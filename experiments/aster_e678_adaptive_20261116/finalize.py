"""CPU finalizer scheduled after all seven GPU jobs reach terminal state."""
from .prepare import OUT,KEYS
from .report import report
from experiments.aster_cr_retrain_paired_v2_20261111.common import read,write

def main():
    smoke=read(OUT/"smoke/qwen/complete.json")
    states={k:read(OUT/"models"/k/"completion.json") for k in KEYS}
    errors={k:read(OUT/"models"/k/"execution.json") for k,v in states.items()
            if not v.get("complete") and not v.get("diagnostics_complete")}
    checks=read(OUT/"test_results.json")
    checks.update(gpu_smoke=smoke,model_experiments=states)
    write(OUT/"test_results.json",checks)
    report()
    write(OUT/"finalization.json",dict(complete=True,
        all_models_finished=all(v.get("complete") or v.get("diagnostics_complete") for v in states.values()),
        scientific_success_count=sum(bool(v.get("complete")) for v in states.values()),
        selection_incomplete_count=sum(v.get("status")=="selection_incomplete" for v in states.values()),
        model_states=states,errors=errors))
if __name__=="__main__":main()
