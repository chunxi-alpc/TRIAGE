"""Wait for existing standard jobs, verify real outputs, then run clinical dev.

This is a one-time dependency chain, not a recurring monitor. It does not
choose parameters from scores or claim that all paper reproduction is complete.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys


def main():
    path = Path(sys.argv[1]).resolve()
    plan = json.loads(path.read_text())
    state_path = path.with_name(path.stem + '_state.json')
    state = {'status': 'waiting_for_existing_jobs', 'pid': os.getpid(),
             'plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'scope': 'Standard implementation milestone followed by train_val engineering development; no SOTA claim',
             'wait_handles': [], 'steps': []}

    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)

    # Atomically claim this plan before waiting or starting any subprocess.
    # A second invocation must not overwrite the first invocation's state.
    with state_path.open('x') as handle:
        json.dump(state, handle, indent=2)
        handle.write('\n')
    handles = []
    active_step = None
    try:
        # Open all process handles now, preventing accidental PID reuse while
        # waiting for another long job. Each handle waits for actual exit.
        for pid in plan['wait_for_pids']:
            try:
                descriptor = os.pidfd_open(int(pid))
            except ProcessLookupError:
                state['wait_handles'].append({'pid': pid, 'state': 'already_exited'})
                continue
            handles.append((pid, descriptor))
            state['wait_handles'].append({'pid': pid, 'state': 'live_handle_opened'})
        save()
        for pid, descriptor in handles:
            event = select.poll()
            event.register(descriptor, select.POLLIN)
            event.poll()
        state['status'] = 'verifying_completed_standard_jobs'
        save()
        for manifest_path in plan['required_run_manifests']:
            manifest = json.loads(Path(manifest_path).read_text())
            if manifest.get('process_state') != 'exited' or manifest.get('exit_code') != 0:
                raise RuntimeError('Prerequisite did not complete successfully: ' + manifest_path)
        for step in plan['verification_steps']:
            step_state = {'name': step['name'], 'status': 'running'}
            active_step = step_state
            state['steps'].append(step_state)
            save()
            report_path = Path(step['report'])
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report.get('status') != 'complete_verified':
                    raise RuntimeError('Existing verification is not complete: ' + str(report_path))
                step_state['execution'] = 'reused_existing_verified_report'
            else:
                with Path(step['log']).open('x') as log:
                    result = subprocess.run(step['command'], cwd=plan['cwd'], stdout=log,
                        stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONHASHSEED='0'))
                step_state['exit_code'] = result.returncode
                if result.returncode:
                    raise RuntimeError('Verification failed: ' + step['name'])
                report = json.loads(report_path.read_text())
                if report.get('status') != 'complete_verified':
                    raise RuntimeError('Verification did not certify its scope: ' + step['name'])
            step_state.update(status='complete_verified', report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
            save()
            active_step = None
        state['status'] = 'clinical_development_running'
        state['clinical_started_at_utc'] = datetime.now(timezone.utc).isoformat()
        state['clinical_scope'] = 'Frozen 16-case train_val development only, not external confirmation'
        state['clinical_status'] = 'running'
        save()
        with Path(plan['clinical_log']).open('x') as log:
            child = subprocess.Popen(plan['clinical_command'], cwd=plan['cwd'], stdout=log,
                stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONHASHSEED='0'))
            state['clinical_pid'] = child.pid
            save()
            code = child.wait()
        state['clinical_exit_code'] = code
        state['clinical_status'] = 'finished_pending_independent_evaluation' if code == 0 else 'failed'
        state['status'] = 'clinical_finished_pending_independent_evaluation' if code == 0 else 'clinical_failed'
        save()
        return code
    except BaseException as error:
        if active_step is not None and active_step['status'] == 'running':
            active_step.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        if state.get('clinical_status') == 'running':
            state['clinical_status'] = 'failed'
        state.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        save()
        raise
    finally:
        for _, descriptor in handles:
            os.close(descriptor)


if __name__ == '__main__':
    raise SystemExit(main())
