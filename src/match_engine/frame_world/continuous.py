"""Continuous-time mixture survival model for post-reception marks."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class ContinuousTimeConfig:
    feature_dim: int = 22
    hidden_dim: int = 64
    components: int = 3
    horizon_s: float = 5.0
    min_time_s: float = 0.025


class ContinuousTimeMarkModel(nn.Module):
    def __init__(self, cfg=ContinuousTimeConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        k = cfg.components
        self.encoder = nn.Sequential(
            nn.Linear(cfg.feature_dim, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.SiLU(),
        )
        self.mixture = nn.Linear(d, k)
        self.location = nn.Linear(d, k)
        self.log_scale = nn.Linear(d, k)
        self.continue_head = nn.Linear(d, 1)

    def forward(self, features):
        state = self.encoder(features)
        weights = self.mixture(state)
        location = self.location(state)
        scale = torch.nn.functional.softplus(self.log_scale(state)) + 0.08
        continuation = self.continue_head(state).squeeze(-1)
        subtype = (features[:, 9] - 0.25) * 10
        return weights, location, scale, continuation, subtype

    def distribution_cdf(self, features, time_s):
        weights, location, scale, _, _ = self(features)
        probability = torch.softmax(weights, 1)
        value = torch.as_tensor(time_s, dtype=features.dtype, device=features.device)
        value = value.expand(len(features)) if value.ndim == 0 else value
        z = (
            torch.log(value.clamp_min(self.cfg.min_time_s))[:, None] - location
        ) / scale
        return (probability * torch.special.ndtr(z)).sum(1)

    def quantile_time(self, features, quantile, iterations=18):
        low = torch.full((len(features),), self.cfg.min_time_s, device=features.device)
        high = torch.full_like(low, self.cfg.horizon_s)
        for _ in range(iterations):
            middle = (low + high) / 2
            cdf = self.distribution_cdf(features, middle)
            high = torch.where(cdf >= quantile, middle, high)
            low = torch.where(cdf < quantile, middle, low)
        raw = (low + high) / 2
        anchor = getattr(self, "time_anchor_s", None)
        shrink = float(getattr(self, "time_shrinkage", 1.0))
        if anchor is None or shrink == 1.0:
            return raw
        raw_median = self._raw_median_time(features, iterations)
        adjusted_median = float(anchor) + shrink * (raw_median - float(anchor))
        return (raw + adjusted_median - raw_median).clamp(
            self.cfg.min_time_s, self.cfg.horizon_s
        )

    def _raw_median_time(self, features, iterations=18):
        weights, location, scale, _, _ = self(features)
        probability = torch.softmax(weights, 1)
        low = torch.full((len(features),), self.cfg.min_time_s, device=features.device)
        high = torch.full_like(low, self.cfg.horizon_s)
        for _ in range(iterations):
            middle = (low + high) / 2
            z = (torch.log(middle[:, None]) - location) / scale
            cdf = (probability * torch.special.ndtr(z)).sum(1)
            high = torch.where(cdf >= 0.5, middle, high)
            low = torch.where(cdf < 0.5, middle, low)
        return (low + high) / 2

    def median_time(self, features, iterations=18):
        raw = self._raw_median_time(features, iterations)
        anchor = getattr(self, "time_anchor_s", None)
        shrink = float(getattr(self, "time_shrinkage", 1.0))
        return raw if anchor is None else float(anchor) + shrink * (raw - float(anchor))

    def predict_mark(self, features, continue_threshold=0.5, subtype_threshold=0.5):
        *_, continuation, subtype = self(features)
        kind = torch.where(
            torch.sigmoid(continuation) >= continue_threshold,
            torch.where(
                torch.sigmoid(subtype) >= subtype_threshold,
                torch.ones_like(continuation, dtype=torch.long),
                torch.zeros_like(continuation, dtype=torch.long),
            ),
            torch.full_like(continuation, 2, dtype=torch.long),
        )
        return self.median_time(features), kind

    def checkpoint(self):
        return {
            "version": "9.0",
            "config": asdict(self.cfg),
            "state_dict": self.state_dict(),
        }


def mixture_survival_loss(weights, location, scale, time_s, observed, min_time_s=0.025):
    time = time_s.clamp_min(min_time_s)
    log_time = torch.log(time)[:, None]
    z = (log_time - location) / scale
    log_mix = torch.log_softmax(weights, 1)
    log_pdf = (
        -log_time - torch.log(scale) - 0.5 * math.log(2 * math.pi) - 0.5 * z.square()
    )
    survival = torch.special.ndtr(-z).clamp_min(1e-12)
    observed_ll = torch.logsumexp(log_mix + log_pdf, 1)
    censored_ll = torch.logsumexp(log_mix + torch.log(survival), 1)
    return -torch.where(observed, observed_ll, censored_ll).mean()


def load_continuous_time(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ContinuousTimeConfig(**payload["config"])
    model = ContinuousTimeMarkModel(cfg)
    model.load_state_dict(payload["state_dict"])
    meta = payload.get("meta", {})
    model.time_anchor_s = meta.get("time_anchor_s", None)
    model.time_shrinkage = meta.get("time_shrinkage", 1.0)
    return model, cfg, meta
