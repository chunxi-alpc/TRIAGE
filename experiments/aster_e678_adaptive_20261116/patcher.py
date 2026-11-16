"""Separate true/sham SQLite caches, with explicit source/pair/head identity."""
import json
import sqlite3
from pathlib import Path
import numpy as np
from .adaptive_m1 import ScopedStore, pair_ids, representatives

class ShamDonorPatcher:
    def __init__(self, root, store, inputs, engine, dev_pairs, identity):
        self.root, self.inputs, self.engine = Path(root), inputs, engine
        representatives(dev_pairs, "dev")
        self.pairs = {(p["source_id"], p["pair_id"]):p for p in dev_pairs}
        self.store = ScopedStore(store, pair_ids(dev_pairs))
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = {}
        self.identity = json.dumps(identity, sort_keys=True, ensure_ascii=False)
        for mode in ("true", "sham"):
            conn = sqlite3.connect(self.root / (mode + ".sqlite"))
            conn.execute("CREATE TABLE IF NOT EXISTS identity (value TEXT NOT NULL)")
            old = conn.execute("SELECT value FROM identity").fetchall()
            if old and old != [(self.identity,)]:
                raise ValueError("Cache model/input/scoring identity changed")
            if not old:
                conn.execute("INSERT INTO identity VALUES (?)", (self.identity,))
            conn.execute("""CREATE TABLE IF NOT EXISTS patches (
                kind TEXT, target_source TEXT, target_pair TEXT, target_left TEXT, target_right TEXT,
                donor_source TEXT, donor_pair TEXT, donor_left TEXT, donor_right TEXT, head INTEGER,
                forward REAL, reverse REAL,
                PRIMARY KEY(kind,target_source,target_pair,target_left,target_right,
                            donor_source,donor_pair,donor_left,donor_right,head))""")
            conn.commit()
            self.db[mode] = conn

    def _validate(self, target, donor, sham):
        for p in (target, donor):
            if p.get("split") != "dev" or self.pairs.get((p["source_id"],p["pair_id"])) != p:
                raise ValueError("Patcher accepts prepared development pairs only")
        if target["kind"] != donor["kind"]:
            raise ValueError("Donor category mismatch")
        if sham and target["source_id"] == donor["source_id"]:
            raise ValueError("Sham donor must have a different source")
        if not sham and target != donor:
            raise ValueError("True intervention requires the paired endpoint")

    def scores(self, target, donor, heads, sham=False):
        self._validate(target, donor, sham)
        conn = self.db["sham" if sham else "true"]
        base = (target["kind"], target["source_id"], target["pair_id"], target["left"], target["right"],
                donor["source_id"], donor["pair_id"], donor["left"], donor["right"])
        fields = ("kind","target_source","target_pair","target_left","target_right",
                  "donor_source","donor_pair","donor_left","donor_right","head")
        query = "SELECT forward,reverse FROM patches WHERE " + " AND ".join(f+"=?" for f in fields)
        cached = {int(h):conn.execute(query, (*base,int(h))).fetchone() for h in heads}
        missing = [int(h) for h in heads if cached[int(h)] is None]
        if missing:
            donor_l = self.store.get(donor["left"])["heads"]
            donor_r = self.store.get(donor["right"])["heads"]
            donor_l = np.asarray(donor_l).reshape(-1, np.asarray(donor_l).shape[-1])
            donor_r = np.asarray(donor_r).reshape(-1, np.asarray(donor_r).shape[-1])
            left = self.engine.prepare(self.inputs[target["left"]]["prompt"])
            right = self.engine.prepare(self.inputs[target["right"]]["prompt"])
            for h in missing:
                f = float(self.engine.run(left, replacements={h:donor_r[h]}, capture=False)["ell"])
                r = float(self.engine.run(right, replacements={h:donor_l[h]}, capture=False)["ell"])
                if not np.isfinite([f,r]).all():
                    raise ValueError("Nonfinite exact patch score")
                conn.execute("INSERT INTO patches VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (*base,h,f,r))
                conn.commit()
                cached[h] = f,r
        return np.asarray([cached[int(h)] for h in heads], float).reshape(-1,2)

    def effects(self, target, donor, heads, sham=False):
        scores = self.scores(target, donor, heads, sham)
        baseline = np.array([self.store.ell(target["left"]),self.store.ell(target["right"])])
        return np.abs(scores - baseline).mean(axis=1)

    def close(self):
        for conn in self.db.values():
            conn.close()
