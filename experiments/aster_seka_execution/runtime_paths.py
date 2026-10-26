"""Preserve virtual-environment executable paths when preparing commands."""
import os
from pathlib import Path

def interpreter_path(value):
    # Resolving bin/python follows its symlink to the system interpreter and loses venv discovery.
    path = Path(value).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError('Executable Python path is unavailable: ' + str(path))
    return path
