"""CARE module-one interface."""
from .care_v6 import CAREModule1V6, run, set_default_v6

def __getattr__(name):
    if name == "Module1":
        from .modeler import Module1
        return Module1
    if name == "run_legacy":
        from .modeler import run as run_legacy
        return run_legacy
    raise AttributeError(name)

__all__ = ["CAREModule1V6", "run", "set_default_v6", "Module1", "run_legacy"]
