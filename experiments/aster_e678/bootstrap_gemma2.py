"""CPU-only resumable Gemma-2-9B-Base download for E8."""
import json
import os
import time
import traceback
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/aster_e678_20261112'
MODELS = json.loads(Path(__file__).with_name('models.json').read_text(encoding='utf-8'))
SPEC = MODELS['gemma2_base']


def write_status(value):
    path = OUT / 'status/gemma2_base_download.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def checkpoint_ready(path):
    path = Path(path)
    index = path / 'model.safetensors.index.json'
    if not index.exists() or not (path / 'config.json').exists() or not (path / 'tokenizer_config.json').exists():
        return False
    try:
        files = set(json.loads(index.read_text())['weight_map'].values())
    except (ValueError, KeyError):
        return False
    return all((path / name).exists() and (path / name).stat().st_size > 0 for name in files)


def main():
    from modelscope import snapshot_download
    started = time.time()
    model_path = Path(SPEC['path'])
    common = dict(model='gemma2_base', source=SPEC['download_source'], path=str(model_path),
                  pid=os.getpid(), started=started, gpu_usage=False)
    if checkpoint_ready(model_path):
        write_status(dict(common, state='complete', reused=True, seconds=0))
        return
    write_status(dict(common, state='downloading'))
    try:
        path = snapshot_download(SPEC['download_source'], local_dir=str(model_path),
                                 ignore_file_pattern=['*.gguf', '*.bin', '*.pth', '*.pt', 'original/*'])
        if not checkpoint_ready(path):
            raise RuntimeError('Download returned without complete checkpoint files')
        write_status(dict(common, state='complete', reused=False, seconds=time.time()-started))
    except BaseException as exc:
        write_status(dict(common, state='failed', error=str(exc), seconds=time.time()-started))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
