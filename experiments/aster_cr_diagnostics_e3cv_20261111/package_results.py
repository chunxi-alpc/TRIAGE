from pathlib import Path
import gzip,json,shutil,time
from .common import *
def main():
    artifacts=[]
    for source in sorted((OUT/"evaluation").glob("*/*.jsonl")):
        dest=source.with_suffix(source.suffix+".gz")
        with source.open("rb") as inp, gzip.open(dest,"wb",compresslevel=6) as compressed:
            shutil.copyfileobj(inp,compressed)
        artifacts.append(dict(path=str(dest.relative_to(OUT)),bytes=dest.stat().st_size))
    write(OUT/"publication_manifest.json",dict(complete=True,experiment_version=VERSION,
        branch="results/aster-cr-diagnostics-e3cv-20261111",historical_commit="d37487a",
        prediction_archives=artifacts,archive_bytes=sum(a["bytes"] for a in artifacts),
        raw_JSONL_policy="Uncompressed originals remain on the server",
        result_files=["REPORT.zh.md","metrics.csv","source_bootstrap.csv","transitions.csv",
            "source_metrics.csv","independent_intercepts.json","m2_subspace_overlap.json",
            "m3_constraint_margins.csv","medpic_option_counts.csv","verification.json",
            "independent_score_verification.json"],
        history_preserved=True))
    print("Prediction archives ready",len(artifacts))
if __name__=="__main__":main()
