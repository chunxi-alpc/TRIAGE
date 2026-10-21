#!/usr/bin/env python3
"""Stop only the identified MedGUIDE process group, preserving all original files."""
import argparse, json, os, signal, subprocess, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
OLD_ROOT = Path("/home/gcx/TRIAGE-worktrees/medguide-full-20261021")
SCRIPT = "experiments/medguide_full_20261021/run.sh"

def snapshot():
    rows = {}
    for p in Path("/proc").glob("[0-9]*"):
        try:
            fields = (p/"stat").read_text().rsplit(")", 1)[1].split()
            rows[int(p.name)] = dict(pid=int(p.name), ppid=int(fields[1]),
                pgid=int(fields[2]), starttime=fields[19], uid=p.stat().st_uid,
                argv=(p/"cmdline").read_bytes().decode().strip("\0").split("\0"),
                cwd=os.readlink(p/"cwd"))
        except (OSError, ValueError):
            pass
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    rows = snapshot()
    seeds = [r for r in rows.values() if r["uid"] == os.getuid()
             and r["cwd"] == str(OLD_ROOT) and SCRIPT in r["argv"]]
    if not seeds:
        print("No matching MedGUIDE run.sh; verify orphaned workers with ps and nvidia-smi separately.")
        return
    groups = {r["pgid"] for r in seeds}
    selected = [r for r in rows.values() if r["pgid"] in groups]
    ids = {r["pid"] for r in selected}
    assert os.getpgrp() not in groups
    assert all(r["uid"] == os.getuid() and r["cwd"] == str(OLD_ROOT)
               and (r["pid"] == r["pgid"] or r["ppid"] in ids) for r in selected)
    assert all(g in rows and SCRIPT in rows[g]["argv"] for g in groups)
    print(json.dumps(selected, indent=2))
    if not args.execute:
        return
    out = HERE/"private"/("stop_" + str(time.time_ns()))
    out.mkdir(parents=True)
    (out/"before.json").write_text(json.dumps(selected, indent=2))
    (out/"gpu_before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    for g in groups:
        # Stop orchestration before terminating workers, so wait cannot reach git push.
        now = snapshot()
        assert now[g]["starttime"] == rows[g]["starttime"]
        os.killpg(g, signal.SIGSTOP)
        os.killpg(g, signal.SIGKILL)
    time.sleep(3)
    remaining = [r for r in snapshot().values() if r["pgid"] in groups]
    (out/"after.json").write_text(json.dumps(remaining, indent=2))
    (out/"gpu_after.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    if remaining:
        raise SystemExit("Stop incomplete; inspect saved evidence")
    print("Identified MedGUIDE groups stopped; files preserved.")

if __name__ == "__main__":
    main()
