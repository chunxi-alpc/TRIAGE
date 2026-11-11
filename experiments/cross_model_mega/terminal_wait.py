"""Wait for the controller's terminal marker, without querying GPUs or running metrics."""
import json,time
from .fit_controlled import OUT
def main():
    path=OUT/'pipeline_status.json'
    while True:
        if path.exists():
            state=json.loads(path.read_text())
            if state.get('state') in ('complete','failed'):
                # The parallel controller writes actual per-shard GPU metadata and TERMINAL_READY.
                print(json.dumps(state,ensure_ascii=False),flush=True)
                return
        time.sleep(60)
if __name__=='__main__':main()
