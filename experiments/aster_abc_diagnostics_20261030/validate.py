"""Run existing ASTER regressions and isolated diagnostic tests on the server."""
import argparse,json,sys,unittest
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 loader=unittest.TestLoader();suite=unittest.TestSuite()
 suite.addTests(loader.discover("tests",pattern="test_aster*.py"))
 for name in ["test_statistics","a.test_a","b.test_b","c.test_c"]:
  suite.addTests(loader.loadTestsFromName("experiments.aster_abc_diagnostics_20261030."+name))
 with (a.output/"tests.log").open("w") as f:
  result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
 summary={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"passed":result.wasSuccessful()}
 (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps(summary))
 if not result.wasSuccessful():sys.exit(1)
if __name__=="__main__":main()
