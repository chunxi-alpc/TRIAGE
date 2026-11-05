"""Focused checks for sharing unfinished annotation work."""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from annotation_queue import claim, finish, pending_ids

class AnnotationQueueTests(unittest.TestCase):
    def test_completed_nodes_are_not_queued_again(self):
        nodes=[{"node_id":name} for name in ("done","old_schema","pending")]
        previous={"done":{"management_extraction_completed":True,"semantic_schema":2},
                  "old_schema":{"management_extraction_completed":True,"semantic_schema":1}}
        self.assertEqual(pending_ids(nodes,previous),["old_schema","pending"])

    def test_two_workers_claim_each_node_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"queue.json"
            ids=[str(i) for i in range(20)]
            path.write_text(json.dumps(dict(pending=ids,claims={},finished={})))
            def work(owner):
                found=[]
                while True:
                    node_id=claim(path,owner)
                    if node_id is None:
                        return found
                    found.append(node_id)
                    finish(path,node_id,"complete")
            with ThreadPoolExecutor(max_workers=2) as executor:
                left=executor.submit(work,"gpu0")
                right=executor.submit(work,"gpu1")
                a,b=left.result(),right.result()
            self.assertFalse(set(a)&set(b))
            self.assertEqual(sorted(a+b,key=int),ids)
            state=json.loads(path.read_text())
            self.assertEqual(len(state["finished"]),20)
            self.assertFalse(state["pending"])
            self.assertFalse(state["claims"])

if __name__=="__main__":
    unittest.main()
