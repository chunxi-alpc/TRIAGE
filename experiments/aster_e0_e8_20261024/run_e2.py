"""Standalone oracle channel comparison. Historical E1 performance is not an engineering gate."""
import argparse
from pathlib import Path
from continue_run import Runner

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--e1',type=Path,help='Historical provenance only; not a performance gate');a=p.parse_args();a.role='main';a.evidence=None
 r=Runner(a);rows=r.ordered('dev',0)
 for i,c in enumerate(rows):
  r.cache={}
  for name,kw in r.oracle_arms(c).items():r.answer(c,'E2',name,oracle=name!='none',**kw)
  if (i+1)%20==0:r.status('E2',i+1,len(rows))
 r.status('complete',len(rows),len(rows))
if __name__=='__main__':main()
