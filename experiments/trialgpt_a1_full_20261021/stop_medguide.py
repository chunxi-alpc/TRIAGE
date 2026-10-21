#!/usr/bin/env python3
"""Safely stop ONLY PR19 MedGUIDE processes by exact project identity.

Default is dry run. Execute on authorized GPU server: python stop_medguide.py --execute
Does not delete logs or unrelated processes.
"""
from __future__ import annotations
import argparse
import os
import signal
import time
from pathlib import Path

PATH_MARKERS = [
    "experiments/medguide_full_20261021/run.sh",
    "experiments/medguide_full_20261021/private/input_",
    "experiments/medguide_full_20261021/qwen_run.py",
    "experiments/medguide_full_20261021/prepare.py",
    "experiments/medguide_full_20261021/summarize.py",
]


def processes():
    items={}
    for x in Path("/proc").glob("[0-9]*"):
        try:
            cmd=(x/"cmdline").read_bytes().replace(b"\x00",b" ").decode(errors="replace")
            status=(x/"status").read_text()
            ppid=int(next(line.split()[1] for line in status.splitlines()
                          if line.startswith("PPid:")))
        except (OSError, ValueError, StopIteration):
            continue
        items[int(x.name)]={"ppid":ppid,"cmd":cmd}
    return items


def relevant_processes(items):
    self_pid=os.getpid()
    seed={pid for pid, row in items.items()
          if pid != self_pid and any(m in row["cmd"] for m in PATH_MARKERS)}
    relevant=set(seed)
    while True:
        children={pid for pid,row in items.items() if row["ppid"] in relevant and pid != self_pid}
        more=children-relevant
        if not more:break
        relevant |= more
    return relevant


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--execute",action="store_true",help="Actually send SIGTERM to matched project job")
    args=ap.parse_args()
    items=processes()
    relevant=relevant_processes(items)
    if not relevant:
        print("No active PR19 MedGUIDE jobs found.")
        return
    roots=[pid for pid in relevant if items[pid]["ppid"] not in relevant]
    print("Matched PR19 process tree; total",len(relevant))
    for pid in sorted(relevant):
        print(pid,"PPID",items[pid]["ppid"],items[pid]["cmd"][:250])
    if not args.execute:
        print("DRY RUN ONLY. Re-run with --execute after confirming the PIDs.")
        return
    # Kill parent orchestration first: old run.sh otherwise commits and pushes
    # even after a worker fails. Kill children immediately afterwards.
    for pid in roots + [p for p in sorted(relevant) if p not in roots]:
        try:
            os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            print("Permission denied, manual action required PID",pid)
    time.sleep(2)
    survivors=relevant_processes(processes())
    if survivors:
        print("STOP INCOMPLETE; manual intervention required for",sorted(survivors))
        raise SystemExit(2)
    print("STOP CONFIRMED: no PR19 MedGUIDE processes remain. Logs preserved.")


if __name__=="__main__":
    main()

