from __future__ import annotations
import heapq
import numpy as np
from src.match_engine.subtick_reception_queue import SubtickReceptionQueue,_QueuedReception
from src.match_engine.state import BallState,CrowdState,MatchAffectiveState,PlayerAffectiveState,RefereeAffectiveState,TeamAffectiveState

def state(possessor="h0"):
    h=PlayerAffectiveState("h0","h0","CM","H",position=np.array([.4,.5]));a=PlayerAffectiveState("a0","a0","CB","A",position=np.array([.6,.5]))
    return MatchAffectiveState(TeamAffectiveState("H",[h]),TeamAffectiveState("A",[a]),RefereeAffectiveState(),CrowdState(),ball=BallState(possessor_id=possessor,possession_team_id="H"))

def queue():
    q=SubtickReceptionQueue.__new__(SubtickReceptionQueue);q.provider="skillcorner";q.blend=1.;q._pending=[];q._sequence=0;q.scheduled=0;q.applied=0;q.cancelled=0;q.cancelled_possession=0;q.cancelled_missing=0;q.cancelled_off_pitch=0;q.mismatch_examples=[];q.skipped_possessor_mismatch=0;q.displacement_sum_m=0.;q.delay_min_s=float("inf");q.delay_max_s=0.;q.delay_sum_s=0.;q.kind_counts={};return q

def event(due,seq):
    return _QueuedReception(due,seq,"h0","a0",{"h0":np.array([.05,0.]),"a0":np.array([-.05,0.])},{"h0":np.array([.01,0.]),"a0":np.array([-.01,0.])},"pass")

def test_subtick_queue_reconciles_due_events_without_rng():
    q=queue();heapq.heappush(q._pending,event(2.,1));heapq.heappush(q._pending,event(2.,0));s=state();q.reconcile(1.9,s);assert q.applied==0 and len(q._pending)==2
    q.reconcile(2.,s);assert q.applied==2 and q.cancelled==0 and len(q._pending)==0;assert np.allclose(s.home.players[0].position,[.5,.5]);assert q.diagnostics()["rng_draws"]==0

def test_subtick_queue_cancels_stale_possession_chain():
    q=queue();heapq.heappush(q._pending,event(1.,0));s=state("a0");q.reconcile(1.,s);assert q.cancelled==1 and q.applied==0
def test_subtick_queue_blends_learned_residual():
    q=queue();q.blend=.5;heapq.heappush(q._pending,event(1.,0));s=state();q.reconcile(1.,s)
    assert np.allclose(s.home.players[0].position,[.425,.5])
    assert np.allclose(s.away.players[0].position,[.575,.5])
