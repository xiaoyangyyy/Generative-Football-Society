#!/usr/bin/env python3
'''Verify strict mirrored seeds and duration convergence.'''
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    short=json.loads((ROOT/'reports/acceptance/strict_mirror_shadow_v99.json').read_text())
    long=json.loads((ROOT/'reports/acceptance/strict_mirror_duration600_v99.json').read_text())
    failures=[]
    for name,report,seconds,seeds in (('short',short,120,8),('long',long,600,4)):
        protocol=report.get('protocol',{})
        if protocol.get('seed_policy')!='unordered_fixture_v1':failures.append(name+'_seed_policy')
        if protocol.get('seconds')!=seconds or protocol.get('seeds')!=seeds:failures.append(name+'_protocol')
        if protocol.get('matched_pairs',0)<100:failures.append(name+'_sample')
    if not short.get('ready') or not all(short.get('gates',{}).values()):failures.append('short_gates')
    allowed={'seed_coverage'}
    failed={key for key,value in long.get('gates',{}).items() if not value}
    if failed!=allowed:failures.append('long_diagnostic_gates')
    effect=long['aggregate']['paired_effects']['possession_home']
    if abs(effect['mean'])>.005 or not effect['ci95'][0]<=0<=effect['ci95'][1]:failures.append('duration_convergence')
    result={'release':'9.9.0-shadow','status':'strict_mirror_duration_converged','short_pairs':short['protocol']['matched_pairs'],'long_pairs':long['protocol']['matched_pairs'],'long_possession_effect':effect,'failures':failures,'ok':not failures}
    print(json.dumps(result,indent=2));return 0 if not failures else 1
if __name__=='__main__':raise SystemExit(main())
