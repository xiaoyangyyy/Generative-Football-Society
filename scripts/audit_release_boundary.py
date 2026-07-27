#!/usr/bin/env python3
'''Fail when a proposed repository release includes private or oversized data.'''
from __future__ import annotations
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MAX_BYTES=95*1024*1024
FORBIDDEN_SUFFIXES=('.checkpoint.json',)

def candidates():
    command=['git','ls-files','--cached','--others','--exclude-standard']
    return sorted(set(subprocess.check_output(command,cwd=ROOT,text=True).splitlines()))

def audit(paths):
    failures=[]
    for rel in paths:
        normalized='/'+rel.replace(chr(92),'/')
        path=ROOT/rel
        if '/data/external/' in normalized and '/raw/' in normalized:failures.append('raw:'+rel)
        if rel.endswith(FORBIDDEN_SUFFIXES):failures.append('checkpoint:'+rel)
        if path.is_file() and path.stat().st_size>MAX_BYTES:failures.append('oversize:'+rel)
    return failures

def main():
    paths=candidates();failures=audit(paths)
    print(f'candidate_files={len(paths)} failures={len(failures)}')
    for failure in failures:print(failure)
    return 0 if not failures else 1
if __name__=='__main__':raise SystemExit(main())
