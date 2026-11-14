"""CPU-only summary after all six independent checkpoint jobs finish."""
from .prepare import OUT, KEYS
from .report import generate
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write
import time
def main():
    missing=[key for key in KEYS if not read(OUT/"models"/key/"completion.json").get("complete")]
    if missing:
        raise RuntimeError("Checkpoint jobs not complete: "+", ".join(missing))
    result=generate()
    if any(value["state"]!="complete" for value in result["E8"].values()):
        raise RuntimeError("Base/IT comparisons are not complete")
    write(OUT/"unified_summary_complete.json",dict(complete=True,
        models=list(KEYS),base_it_comparisons=["gemma2","qwen"],
        report=str(OUT/"REPORT.zh.md"),completed_at=time.strftime("%Y-%m-%d %H:%M:%S")))
if __name__=="__main__":
    main()
