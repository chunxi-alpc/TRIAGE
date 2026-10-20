"""CARE module-one interface.

For new module-two research, create_module1() returns the frozen semantic A1.
Existing run(), create_module1_v6() and the V6 pipeline remain compatible.
"""
from .care_v6 import CAREModule1V6, create_module1_v6, run, set_default_v6, validate_v6_input
from .semantic_a1 import CAREModule1A1, create_module1_a1, signal_targets

create_module1 = create_module1_a1
run_v6 = run


def __getattr__(name):
    if name in {"Module1", "LegacyModule1"}:
        from .modeler import Module1
        return Module1
    if name == "run_legacy":
        from .modeler import run as run_legacy
        return run_legacy
    raise AttributeError(name)


__all__ = [
    "CAREModule1A1", "create_module1", "create_module1_a1", "signal_targets",
    "CAREModule1V6", "create_module1_v6", "run", "run_v6",
    "set_default_v6", "validate_v6_input", "LegacyModule1", "Module1", "run_legacy",
]
