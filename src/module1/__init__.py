"""CARE module-one defaults to Qwen-free V5 with the existing pipeline contract.

Both ``Module1(...).run(guideline, patient, question)`` and the convenience ``run``
use TrialMatchAI + BioLinkBERT-MedNLI. The old implementation is accessible only
through the explicit ``LegacyModule1`` or ``run_legacy`` names.
"""
from .care_v5 import CAREModule1V5, run, set_default_v5
from .compat_v5 import Module1


def __getattr__(name):
    if name == "LegacyModule1":
        from .modeler import Module1 as LegacyModule1
        return LegacyModule1
    if name == "run_legacy":
        from .modeler import run as run_legacy
        return run_legacy
    raise AttributeError(name)


__all__ = ["CAREModule1V5", "Module1", "run", "set_default_v5", "LegacyModule1", "run_legacy"]
