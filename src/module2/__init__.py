"""Offline applicability learning; legacy localization remains lazily available.

Data validation and NumPy statistics must not require PyTorch or a model load.
"""
from importlib import import_module

_EXPORTS = {
    "build_calibration_pairs": (".data", "build_calibration_pairs"),
    "build_cached_calibration_pairs": (".data", "build_cached_calibration_pairs"),
    "locate_heads": (".localizer", "locate_heads"),
    "scale_query_heads": (".localizer", "scale_query_heads"),
    "build_minimal_pairs": (".minimal_pairs", "build_minimal_pairs"),
    "fit_operators": (".mechanism", "fit_operators"),
    "evaluate_direction": (".mechanism", "evaluate_direction"),
    "select_heads": (".mechanism", "select_heads"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
