"""CARE module-two incremental N1--N5 experiment package.

The package deliberately keeps data construction, truth evaluation, leakage
auditing, A1 joining, and model execution separate.  The generators create
synthetic mechanism probes; they are not clinical recommendations.
"""

from .oracle import evaluate, enumerate_completions, action_set, decisive_attributes

__all__ = ["evaluate", "enumerate_completions", "action_set", "decisive_attributes"]
