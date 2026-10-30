"""Execute the complete isolated CPU pipeline from a concrete configuration."""
import argparse,json,subprocess,sys,time
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json"));p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 config=json.loads(a.config.read_text());src=Path(config["official_readonly_root"]).resolve();out=a.output.resolve()
 if out==src or src in out.parents:raise ValueError("Diagnostic output must be outside the frozen official run")
 out.mkdir(parents=True,exist_ok=False);start=time.time()
 module="experiments.aster_abc_diagnostics_20261030."
 commands=[
 ("tests",[module+"validate","--output",str(out/"tests")]),
 ("sources",[module+"leakage","--run",str(src),"--out",str(out/"d")]),
 ("direction",[module+"a.cpu","--base",str(src),"--output",str(out/"a/cpu")]),
 ("candidates",[module+"a.candidates","--base",str(src),"--output",str(out/"a/candidates")]),
 ("saved_query",[module+"b.saved_responses","--input-run",str(src),"--output",str(out/"b/saved_responses")]),
 ("routing",[module+"c.audit","--run",str(src),"--phase","confirm","--output",str(out/"c/cpu")]),
 ("risk",[module+"c.risk","--run",str(src),"--output",str(out/"c/risk")])]
 records=[]
 for stage,arguments in commands:
  cmd=[sys.executable,"-m",*arguments]
  with (out/(stage+".log")).open("x") as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
  records.append({"stage":stage,"command":cmd,"exit_code":code})
  manifest={"status":"failed" if code else "running","stages":records,"configuration":config,"elapsed_seconds":time.time()-start}
  (out/"pipeline.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
  if code:raise SystemExit(code)
 manifest["status"]="complete"
 (out/"pipeline.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps({"status":"complete","stages":len(records),"output":str(out),"elapsed_seconds":time.time()-start}))
if __name__=="__main__":main()
