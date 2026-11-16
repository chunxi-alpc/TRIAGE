import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111.common import write, rows
from experiments.aster_e678_adaptive_20261116.cooperative_extract import extract

class Array:
    def numpy(self):
        return np.ones((2, 3), dtype=np.float32)

class Engine:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()
    def prepare(self, prompt):
        return prompt
    def run(self, prompt, capture, grad):
        with self.lock:
            self.calls.append((prompt, grad))
        time.sleep(.002)
        result = dict(ell=1., heads=Array())
        if grad:
            result["grads"] = Array()
        return result

class CooperativeTests(unittest.TestCase):
    def test_two_workers_once_resume_and_gradient_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ids = [str(i) for i in range(40)]
            write(root/"stage_ids.json", dict(fit=ids))
            write(root/"gradient_ids.json", ids[:4])
            inputs = {uid:dict(prompt=uid) for uid in ids}
            engine = Engine()
            with ThreadPoolExecutor(2) as pool:
                a = pool.submit(extract, root, "fit", inputs, engine, False, "main")
                b = pool.submit(extract, root, "fit", inputs, engine, True, "helper")
                a.result()
                b.result()
            self.assertEqual(len(engine.calls), 40)
            self.assertEqual(set(x for x,g in engine.calls if g), set(ids[:4]))
            records = rows(root/"cache/scores_fit.jsonl")
            self.assertEqual(len(records),40)
            self.assertEqual(len({r["uid"] for r in records}),40)
            extract(root,"fit",inputs,engine)
            self.assertEqual(len(engine.calls),40)
            for r in records:
                with np.load(r["feature"]) as data:
                    self.assertEqual("grads" in data, r["uid"] in ids[:4])

    def test_test_extraction_rejected_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,"parameter freeze"):
                extract(Path(td),"test",{},Engine())

if __name__ == "__main__":
    unittest.main()
