"""PyTorch latent world model: encoder → GRU/Transformer transition → decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.match_engine.world_model.action_codec import ACTION_DIM
from src.match_engine.world_model.config import WorldModelConfig
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.uncertainty import (
    ensemble_uncertainty_decomposition,
)

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:
    _TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass
class WorldModelOutput:
    next_obs: np.ndarray
    pass_success: float
    progress_delta: float
    shot_goal_prob: float
    latent: np.ndarray
    uncertainty: float = 0.0
    epistemic_uncertainty: float | None = None
    aleatoric_uncertainty: float | None = None
    uncertainty_source: str = "legacy_total_as_epistemic_proxy"
    uncertainty_components: dict | None = None
    uncertainty_samples: dict | None = None

    def __post_init__(self) -> None:
        if self.epistemic_uncertainty is None:
            self.epistemic_uncertainty = float(self.uncertainty)
        if self.aleatoric_uncertainty is None:
            self.aleatoric_uncertainty = 0.0

if _TORCH:

    class LatentWorldModel(nn.Module):
        def __init__(self, cfg: WorldModelConfig):
            super().__init__()
            self.cfg = cfg
            self.checkpoint_version = 7
            self.transition_ensemble_trained = True
            d = cfg.latent_dim
            h = cfg.hidden_dim
            self.encoder = nn.Sequential(
                nn.Linear(OBS_DIM, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Linear(h, d),
            )
            self.ln_latent = nn.LayerNorm(d)
            component = max(16, d // 2)
            attn_heads = 4 if component % 4 == 0 else (2 if component % 2 == 0 else 1)
            self.grid_encoder = nn.Sequential(
                nn.Conv2d(5, 16, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Flatten(),
                nn.Linear(16 * 8 * 5, component),
            )
            self.player_token = nn.Linear(4, component)
            self.player_slot_embedding = nn.Embedding(22, component)
            self.player_team_embedding = nn.Embedding(2, component)
            self.graph_message = nn.Sequential(
                nn.Linear(component * 2 + 6, component),
                nn.SiLU(),
                nn.Linear(component, component),
            )
            self.graph_update = nn.Sequential(
                nn.Linear(component * 2, component),
                nn.LayerNorm(component),
                nn.SiLU(),
            )
            player_layer = nn.TransformerEncoderLayer(
                d_model=component,
                nhead=attn_heads,
                dim_feedforward=max(64, component * 2),
                batch_first=True,
                dropout=0.05,
                activation="gelu",
            )
            self.player_encoder = nn.TransformerEncoder(player_layer, num_layers=1)
            self.context_encoder = nn.Sequential(
                nn.Linear(19, component),
                nn.LayerNorm(component),
                nn.SiLU(),
            )
            self.structured_fuse = nn.Sequential(
                nn.Linear(component * 3, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Linear(h, d),
            )
            self.action_embed = nn.Linear(ACTION_DIM, d)
            kind = (cfg.transition_type or "gru").lower()
            if kind == "transformer":
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=max(1, cfg.transformer_heads),
                    dim_feedforward=h,
                    batch_first=True,
                    dropout=0.05,
                )
                self.sequence_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.transformer_layers)
                self.extra_sequence_encoders = nn.ModuleList()
                for _ in range(cfg.transition_ensemble_size - 1):
                    member_layer = nn.TransformerEncoderLayer(
                        d_model=d,
                        nhead=max(1, cfg.transformer_heads),
                        dim_feedforward=h,
                        batch_first=True,
                        dropout=0.05,
                    )
                    self.extra_sequence_encoders.append(
                        nn.TransformerEncoder(
                            member_layer, num_layers=cfg.transformer_layers,
                        )
                    )
                self.extra_grus = nn.ModuleList()
                self.transition_kind = "transformer"
            else:
                self.gru = nn.GRU(d + ACTION_DIM, d, batch_first=True)
                self.extra_grus = nn.ModuleList(
                    nn.GRU(d + ACTION_DIM, d, batch_first=True)
                    for _ in range(cfg.transition_ensemble_size - 1)
                )
                self.extra_sequence_encoders = nn.ModuleList()
                self.transition_kind = "gru"

            self.decoder = nn.Sequential(
                nn.Linear(d, h),
                nn.ReLU(),
                nn.Linear(h, OBS_DIM),
            )
            self.head_pass = nn.Linear(d, 1)
            self.head_xg = nn.Linear(d, 1)
            self.head_shot = nn.Linear(d, 1)
            extra_heads = max(0, int(cfg.ensemble_size) - 1)
            self.extra_pass_heads = nn.ModuleList(nn.Linear(d, 1) for _ in range(extra_heads))
            self.extra_xg_heads = nn.ModuleList(nn.Linear(d, 1) for _ in range(extra_heads))
            self.extra_shot_heads = nn.ModuleList(nn.Linear(d, 1) for _ in range(extra_heads))
            nn.init.zeros_(self.head_shot.weight)
            nn.init.zeros_(self.head_shot.bias)
            for head in self.extra_shot_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

        def decode_next(self, obs: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
            decoded = self.decoder(z_next)
            if self.checkpoint_version >= 3:
                delta = torch.tanh(decoded) * float(self.cfg.residual_scale)
                return torch.clamp(obs + delta, 0.0, 1.0)
            return decoded

        def encode(self, obs: torch.Tensor) -> torch.Tensor:
            if self.checkpoint_version < 4:
                return self.ln_latent(self.encoder(obs))
            grid = obs[:, :200].reshape(-1, 5, 8, 5)
            players = obs[:, 210:298].reshape(-1, 22, 4)
            context = torch.cat([obs[:, 200:210], obs[:, 298:307]], dim=-1)
            grid_z = self.grid_encoder(grid)
            slots = torch.arange(22, device=obs.device).unsqueeze(0)
            teams = torch.cat(
                [
                    torch.zeros(11, dtype=torch.long, device=obs.device),
                    torch.ones(11, dtype=torch.long, device=obs.device),
                ]
            ).unsqueeze(0)
            player_tokens = (
                self.player_token(players)
                + self.player_slot_embedding(slots)
                + self.player_team_embedding(teams)
            )
            if self.checkpoint_version >= 5:
                positions = players[:, :, :2]
                velocities = players[:, :, 2:4]
                delta = positions[:, None, :, :] - positions[:, :, None, :]
                rel_velocity = velocities[:, None, :, :] - velocities[:, :, None, :]
                distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                same_team = (
                    teams[:, :, None] == teams[:, None, :]
                ).to(player_tokens.dtype).unsqueeze(-1).expand(players.shape[0], -1, -1, -1)
                edge = torch.cat([delta, distance, rel_velocity, same_team], dim=-1)
                source = player_tokens[:, :, None, :].expand(-1, -1, 22, -1)
                target = player_tokens[:, None, :, :].expand(-1, 22, -1, -1)
                messages = self.graph_message(torch.cat([source, target, edge], dim=-1))
                mask = (~torch.eye(22, dtype=torch.bool, device=obs.device)).view(1, 22, 22, 1)
                messages = (messages * mask).sum(dim=2) / 21.0
                player_tokens = player_tokens + self.graph_update(
                    torch.cat([player_tokens, messages], dim=-1)
                )
            player_z = self.player_encoder(player_tokens).mean(dim=1)
            context_z = self.context_encoder(context)
            return self.ln_latent(self.structured_fuse(torch.cat([grid_z, player_z, context_z], dim=-1)))

        def outcome_ensemble(self, z: torch.Tensor, action: Optional[torch.Tensor] = None):
            if self.checkpoint_version < 4:
                return (
                    self.head_pass(z).unsqueeze(0),
                    self.head_xg(z).unsqueeze(0),
                    self.head_shot(z).unsqueeze(0),
                )
            pass_logits = torch.stack([self.head_pass(z), *[head(z) for head in self.extra_pass_heads]], dim=0)
            progress = torch.stack([self.head_xg(z), *[head(z) for head in self.extra_xg_heads]], dim=0)
            shot_logits = torch.stack([self.head_shot(z), *[head(z) for head in self.extra_shot_heads]], dim=0)
            if self.checkpoint_version >= 6 and action is not None:
                pass_logits = 0.45 * torch.tanh(pass_logits)
                pass_prior = action[:, 13:14].clamp(0.02, 0.98)
                pass_logits = pass_logits + torch.logit(pass_prior).unsqueeze(0)
            if self.checkpoint_version >= 5 and action is not None:
                shot_logits = 0.35 * torch.tanh(shot_logits)
                xg_prior = action[:, 13:14].clamp(0.01, 0.78)
                prior_logit = torch.logit(xg_prior).unsqueeze(0)
                shot_logits = shot_logits + prior_logit
            return pass_logits, progress, shot_logits

        @property
        def transition_member_count(self) -> int:
            return 1 + (
                len(self.extra_sequence_encoders)
                if self.transition_kind == "transformer"
                else len(self.extra_grus)
            )

        def _transition_member_step(
            self,
            z: torch.Tensor,
            action: torch.Tensor,
            h: Optional[torch.Tensor],
            member_index: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if self.transition_kind == "transformer":
                encoder = (
                    self.sequence_encoder
                    if member_index == 0
                    else self.extra_sequence_encoders[member_index - 1]
                )
                za = z + self.action_embed(action)
                seq = torch.stack([z, za], dim=1)
                if h is not None and h.dim() == 3:
                    seq = torch.cat([h, seq], dim=1)
                out = encoder(seq)
                z_next = out[:, -1, :]
                h_new = out[:, -2:, :]
                return z_next, h_new

            inp = torch.cat([z, action], dim=-1).unsqueeze(1)
            recurrent = (
                self.gru
                if member_index == 0 else self.extra_grus[member_index - 1]
            )
            if h is None:
                out, h_new = recurrent(inp)
            else:
                out, h_new = recurrent(inp, h)
            return out.squeeze(1), h_new

        def _transition_step(
            self,
            z: torch.Tensor,
            action: torch.Tensor,
            h: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Primary-member compatibility path."""
            return self._transition_member_step(z, action, h, 0)

        def transition_ensemble_step(
            self,
            z: torch.Tensor,
            action: torch.Tensor,
            h_members: Optional[list[Optional[torch.Tensor]]] = None,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            hidden = h_members or [None] * self.transition_member_count
            outputs = [
                self._transition_member_step(z, action, hidden[index], index)
                for index in range(self.transition_member_count)
            ]
            return (
                torch.stack([item[0] for item in outputs], dim=0),
                [item[1] for item in outputs],
            )

        def transition_predictions(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return member-specific latent and decoded next-state forecasts."""
            z = self.encode(obs)
            z_members, _ = self.transition_ensemble_step(z, action)
            next_members = torch.stack([
                self.decode_next(obs, member) for member in z_members
            ], dim=0)
            return z_members, next_members

        def initialize_transition_ensemble_from_primary(self) -> None:
            """Make legacy checkpoint expansion exact rather than random."""
            members = (
                self.extra_sequence_encoders
                if self.transition_kind == "transformer" else self.extra_grus
            )
            primary = (
                self.sequence_encoder
                if self.transition_kind == "transformer" else self.gru
            )
            for member in members:
                member.load_state_dict(primary.state_dict())
            self.transition_ensemble_trained = False

        def forward(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            h: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            z = self.encode(obs)
            z_next, h_new = self._transition_step(z, action, h)
            next_obs = self.decode_next(obs, z_next)
            pass_all, xg_all, shot_all = self.outcome_ensemble(z_next, action)
            pass_logit = pass_all.mean(dim=0)
            xg_delta = xg_all.mean(dim=0)
            shot_logit = shot_all.mean(dim=0)
            return z_next, next_obs, pass_logit, xg_delta, shot_logit, h_new

        def imagine(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            *,
            steps: int = 1,
            h: Optional[np.ndarray] = None,
        ) -> WorldModelOutput:
            self.eval()
            with torch.no_grad():
                initial_obs = torch.from_numpy(obs).float().unsqueeze(0)
                act = torch.from_numpy(action).float().unsqueeze(0)
                member_obs = [
                    initial_obs.clone() for _ in range(self.transition_member_count)
                ]
                member_z = [self.encode(value) for value in member_obs]
                member_hidden: list[Optional[torch.Tensor]] = [
                    None for _ in range(self.transition_member_count)
                ]
                if h is not None:
                    initial_hidden = torch.from_numpy(h).float()
                    if initial_hidden.dim() == 2:
                        initial_hidden = initial_hidden.unsqueeze(0)
                    member_hidden = [
                        initial_hidden.clone()
                        for _ in range(self.transition_member_count)
                    ]
                for _ in range(max(1, steps)):
                    next_obs = []
                    next_z = []
                    next_hidden = []
                    for index in range(self.transition_member_count):
                        transitioned, hidden = self._transition_member_step(
                            member_z[index], act, member_hidden[index], index,
                        )
                        decoded = self.decode_next(
                            member_obs[index], transitioned,
                        )
                        next_obs.append(decoded)
                        next_z.append(self.encode(decoded))
                        next_hidden.append(hidden)
                    member_obs = next_obs
                    member_z = next_z
                    member_hidden = next_hidden
                next_obs_members = torch.stack(member_obs, dim=0)
                next_obs_t = next_obs_members.mean(dim=0)
                outcome_members = [
                    self.outcome_ensemble(z_value, act)
                    for z_value in member_z
                ]
                pass_logits = torch.stack([
                    item[0] for item in outcome_members
                ], dim=0)
                pass_probs = torch.sigmoid(pass_logits)
                progress_vals = torch.stack([
                    item[1] for item in outcome_members
                ], dim=0)
                shot_logits = torch.stack([
                    item[2] for item in outcome_members
                ], dim=0)
                shot_probs = torch.sigmoid(shot_logits)
                ps = pass_probs.mean().item()
                xg = progress_vals.mean().item()
                sg = shot_probs.mean().item()
                decomposition = ensemble_uncertainty_decomposition(
                    pass_probs.detach().cpu().numpy(),
                    shot_probs.detach().cpu().numpy(),
                    progress_vals.detach().cpu().numpy(),
                    transition_state_samples=(
                        next_obs_members.detach().cpu().numpy()
                    ),
                    transition_ensemble_trained=(
                        self.transition_ensemble_trained
                    ),
                )
                hidden_mean = torch.stack([
                    value for value in member_hidden if value is not None
                ], dim=0).mean(dim=0)
                h_out = hidden_mean.squeeze(0).detach().cpu().numpy()
            return WorldModelOutput(
                next_obs=(
                    next_obs_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
                ),
                pass_success=float(ps),
                progress_delta=float(xg),
                shot_goal_prob=float(sg),
                latent=h_out.astype(np.float32),
                uncertainty=float(decomposition["total_uncertainty"]),
                epistemic_uncertainty=float(
                    decomposition["epistemic_uncertainty"]
                ),
                aleatoric_uncertainty=float(
                    decomposition["aleatoric_uncertainty"]
                ),
                uncertainty_source=str(decomposition["source"]),
                uncertainty_components=dict(decomposition["components"]),
                uncertainty_samples={
                    "pass_logits": pass_logits.detach().cpu().numpy(),
                    "shot_logits": shot_logits.detach().cpu().numpy(),
                    "progress": progress_vals.detach().cpu().numpy(),
                    "transition_states": (
                        next_obs_members.detach().cpu().numpy()
                    ),
                    "transition_ensemble_trained": (
                        self.transition_ensemble_trained
                    ),
                },
            )

else:

    class LatentWorldModel:  # type: ignore
        def __init__(self, cfg: WorldModelConfig):
            raise RuntimeError(
                "PyTorch is required for the latent world model. Install: pip install torch"
            )


def build_model(cfg: WorldModelConfig | None = None) -> "LatentWorldModel":
    if not _TORCH:
        raise RuntimeError("PyTorch is required. pip install torch")
    return LatentWorldModel(cfg or WorldModelConfig.from_env())


def save_checkpoint(
    path: str,
    model: "LatentWorldModel",
    cfg: WorldModelConfig,
    meta: Dict | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "cfg": cfg.__dict__,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "meta": meta or {},
        "version": 7,
    }
    torch.save(payload, path)


def load_checkpoint(path: str) -> Tuple["LatentWorldModel", WorldModelConfig, Dict]:
    if not _TORCH:
        raise RuntimeError("PyTorch is required. pip install torch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = WorldModelConfig(**payload.get("cfg", {}))
    model = LatentWorldModel(cfg)
    model.checkpoint_version = int(payload.get("version", 2))
    if model.checkpoint_version < 4:
        cfg.imagination_steps = 1
    try:
        if model.checkpoint_version >= 7:
            model.load_state_dict(payload["state_dict"], strict=True)
            model.transition_ensemble_trained = True
        else:
            incompatible = model.load_state_dict(
                payload["state_dict"], strict=False,
            )
            if model.checkpoint_version >= 5:
                allowed_prefixes = (
                    "extra_grus.", "extra_sequence_encoders.",
                )
                unexpected_missing = [
                    key for key in incompatible.missing_keys
                    if not key.startswith(allowed_prefixes)
                ]
                if unexpected_missing or incompatible.unexpected_keys:
                    raise RuntimeError(
                        "unexpected legacy state mismatch: "
                        f"missing={unexpected_missing}, "
                        f"unexpected={incompatible.unexpected_keys}"
                    )
            model.initialize_transition_ensemble_from_primary()
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint incompatible ({exc}). Retrain: python scripts/ensure_world_model.py"
        ) from exc
    model.eval()
    return model, cfg, payload.get("meta", {})
