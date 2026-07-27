#!/usr/bin/env python3
'''Verify v9.8 large-shadow evidence without treating it as deployment proof.'''
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    report=json.loads((ROOT/'reports/acceptance/large_shadow_v98.json').read_text())
    protocol=report.get('protocol',{});agg=report.get('aggregate',{});failures=[]
    if protocol.get('matched_pairs',0)<300 or len(protocol.get('teams',[]))<12:failures.append('sample_coverage')
    if protocol.get('seeds_per_fixture',0)<8 or protocol.get('seconds')!=120:failures.append('protocol')
    if not report.get('ready') or not all(report.get('gates',{}).values()):failures.append('gates')
    if agg.get('scheduled')!=agg.get('applied') or agg.get('cancelled')!=0:failures.append('reconciliation')
    if agg.get('runtime',{}).get('treatment_overhead_ratio',99)>1.10:failures.append('performance')
    if not report.get('deployment_unchanged'):failures.append('deployment_boundary')
    result={'release':'9.8.0-shadow','status':'large_sample_shadow','matched_pairs':protocol.get('matched_pairs'),'failures':failures,'ok':not failures}
    print(json.dumps(result,indent=2));return 0 if not failures else 1
if __name__=='__main__':raise SystemExit(main())
