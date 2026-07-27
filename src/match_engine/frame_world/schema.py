"""Provider-neutral frame sequence contract with explicit visibility."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MAX_TEAM_PLAYERS=16
ENTITY_COUNT=MAX_TEAM_PLAYERS*2+1
BALL_INDEX=ENTITY_COUNT-1

@dataclass(frozen=True)
class FrameSequence:
    provider:str
    match_id:str
    timestamps:np.ndarray
    positions:np.ndarray
    velocities:np.ndarray
    visible:np.ndarray
    teams:np.ndarray
    possession:np.ndarray
    def __post_init__(self):
        length=len(self.timestamps); expected=(length,ENTITY_COUNT,2)
        if self.positions.shape!=expected or self.velocities.shape!=expected or self.visible.shape!=(length,ENTITY_COUNT): raise ValueError("invalid frame sequence shapes")
        if self.teams.shape!=(ENTITY_COUNT,) or self.possession.shape!=(length,): raise ValueError("invalid frame metadata shapes")
        if length>1 and not np.all(np.diff(self.timestamps)>0): raise ValueError("timestamps must be strictly increasing")
        observed=self.positions[self.visible.astype(bool)]
        if len(observed) and (not np.isfinite(observed).all() or observed.min()<-0.15 or observed.max()>1.15): raise ValueError("observed positions exceed the normalized pitch buffer")
        if not np.isfinite(self.positions).all() or not np.isfinite(self.velocities).all(): raise ValueError("frame tensors must remain finite behind visibility masks")

def constant_velocity_baseline(positions:np.ndarray,velocities:np.ndarray,horizon_s:float)->np.ndarray:
    return np.clip(np.asarray(positions,float)+float(horizon_s)*np.asarray(velocities,float),-0.15,1.15)

def save_sequence(path,sequence:FrameSequence)->None:
    np.savez_compressed(path,provider=sequence.provider,match_id=sequence.match_id,timestamps=sequence.timestamps.astype(np.float32),positions=sequence.positions.astype(np.float16),velocities=sequence.velocities.astype(np.float16),visible=sequence.visible.astype(np.uint8),teams=sequence.teams.astype(np.int8),possession=sequence.possession.astype(np.int8))

def load_sequence(path)->FrameSequence:
    with np.load(path,allow_pickle=False) as data:
        return FrameSequence(str(data["provider"]),str(data["match_id"]),data["timestamps"].astype(np.float32),data["positions"].astype(np.float32),data["velocities"].astype(np.float32),data["visible"].astype(bool),data["teams"].astype(np.int8),data["possession"].astype(np.int8))
