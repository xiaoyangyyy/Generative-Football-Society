#!/usr/bin/env python3
"""Strict leave-one-provider-out evaluation for the joint pass model."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.match_engine.joint_pass_model import PASS_FEATURES, fit_joint_pass_model
from scripts.train_joint_pass_v7 import _assign_splits, _completion_metrics, _read_many, _receiver_metrics

def load_data():
    sportec=_read_many("data/external/sportec/derived/*_passes.csv"); sportec.loc[sportec.is_receiver.ne(1),"completion"]=np.nan
    statsbomb=pd.read_csv(ROOT/"data/external/statsbomb360/derived/passes.csv"); statsbomb["is_receiver"]=1
    skill_pass=pd.read_csv(ROOT/"data/external/skillcorner/derived/passes.csv"); skill_pass["is_receiver"]=1
    skill_receiver=pd.read_csv(ROOT/"data/external/skillcorner/derived/receiver_candidates.csv"); skill_receiver["completion"]=np.nan; skill_receiver["physics_prior"]=.5
    metrica=_read_many("data/external/metrica/pass_candidates_game*.csv"); metrica["provider"]="metrica"; metrica["match_id"]="game"+metrica.game.astype(str); metrica["completion"]=np.nan; metrica["physics_prior"]=.5
    data=pd.concat([sportec,statsbomb,skill_pass,skill_receiver,metrica],ignore_index=True,sort=False)
    data["match_id"]=data.match_id.astype(str); data["event_id"]=data.provider.astype(str)+":"+data.event_id.astype(str)
    for feature in PASS_FEATURES:
        if feature not in data: data[feature]=np.nan
    return data

def predict(model,frame,score_receivers=False):
    values=frame[list(PASS_FEATURES)].to_numpy(float)
    result=[model.predict(row,physics_prior=prior) for row,prior in zip(values,frame.physics_prior.fillna(.5))]
    receiver=np.zeros(len(frame),float)
    if score_receivers:
        for indices in frame.groupby("event_id",sort=False).indices.values():
            if len(indices)>=2: receiver[indices]=model.score_receiver_candidates(values[indices])
    return receiver,np.array([r.completion_probability for r in result])

def main():
    data=load_data(); reports={}
    for held_out in sorted(data.provider.unique()):
        source=data[data.provider.ne(held_out)].copy(); source["split"]=_assign_splits(source)
        train=source[source.split.eq("train")]
        model=fit_joint_pass_model(train[list(PASS_FEATURES)].to_numpy(),train.event_id.to_numpy(),train.is_receiver.fillna(0).to_numpy(),train.completion.to_numpy(),train.physics_prior.fillna(.5).to_numpy(),steps=50,learning_rate=.025)
        dev=source[source.split.eq("dev")&source.completion.notna()&source.is_receiver.eq(1)].reset_index(drop=True)
        _,prob=predict(model,dev); thresholds=np.linspace(.05,.95,181); threshold=max(thresholds,key=lambda t:_completion_metrics(dev,prob,t)["balanced_accuracy"])
        test=data[data.provider.eq(held_out)&data.completion.notna()&data.is_receiver.eq(1)].reset_index(drop=True)
        completion=None
        if len(test):
            _,prob=predict(model,test); completion=_completion_metrics(test,prob,float(threshold))
            completion["auc_delta_vs_physics"]=completion["auc"]-completion["physics_prior_auc"]
            completion["positives"]=int(test.completion.sum()); completion["negatives"]=int(len(test)-test.completion.sum())
            completion["evidence_sufficient"]=completion["positives"]>=80 and completion["negatives"]>=80
        rank=data[data.provider.eq(held_out)&data.is_receiver.notna()].reset_index(drop=True)
        receiver=None
        if len(rank):
            score,_=predict(model,rank,score_receivers=True); candidate=_receiver_metrics(rank,score)
            if candidate["events"]: receiver=candidate
        reports[held_out]={"train_providers":sorted(source.provider.unique()),"calibration_policy":"none for unseen provider; retain physics anchor","completion":completion,"receiver":receiver}
        print(json.dumps({held_out:reports[held_out]},indent=2),flush=True)
    completion_folds=[v["completion"] for v in reports.values() if v["completion"] and v["completion"]["evidence_sufficient"]]
    skill=reports["skillcorner"]
    completion_ready=all(v["auc_delta_vs_physics"]>=-.02 and v["brier"]<=v["physics_prior_brier"]+.01 for v in completion_folds)
    skillcorner_ready=completion_ready and skill["receiver"] is not None and skill["receiver"]["mrr"]>=.35
    receiver_folds=[v["receiver"] for v in reports.values() if v["receiver"]]
    all_provider_ready=skillcorner_ready and all(v["mrr"]>=.35 for v in receiver_folds)
    report={"version":"7.0.0-candidate","policy":"strict leave-one-provider-out; held provider excluded from fit and calibration","folds":reports,"completion_lodo_ready":bool(completion_ready),"skillcorner_third_provider_ready":bool(skillcorner_ready),"all_provider_lodo_ready":bool(all_provider_ready)}
    target=ROOT/"reports/acceptance/joint_pass_lodo_v7.json"; target.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"skillcorner_third_provider_ready":skillcorner_ready,"all_provider_lodo_ready":all_provider_ready},indent=2)); return 0 if skillcorner_ready else 1
if __name__=="__main__": raise SystemExit(main())
