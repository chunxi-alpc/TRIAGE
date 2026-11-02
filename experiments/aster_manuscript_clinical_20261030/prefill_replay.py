"""Replay identical full-prompt forwards inside one unedited scoring context.

The original Hugging Face generate/sampling loop and RNG calls remain unchanged.
Only empty-cache, batch-one, multi-token prefill forwards are eligible. One prompt
is retained; every return owns a fresh deep copy of the unmodified cache. Never
use this context across model/cache edits or concurrent calls to the same model.
"""
from contextlib import contextmanager
import copy
import functools
import inspect
import time

import torch
from transformers.cache_utils import DynamicCache


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device, type)):
        return value
    if isinstance(value, tuple):
        return tuple(_snapshot(v) for v in value)
    if isinstance(value, list):
        return [_snapshot(v) for v in value]
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    raise TypeError('unsupported forward argument for exact prefill replay')


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and a.device == b.device and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _empty_cache_signature(cache):
    if cache is None:
        return ('none',)
    # Reject sliding/offloaded/static/custom cache behavior. Standard full-attention
    # DynamicCache is the registered execution interface for this experiment.
    if type(cache) is not DynamicCache or getattr(cache, 'offloading', False):
        return None
    if int(cache.get_seq_length()) != 0:
        return None
    layers = getattr(cache, 'layers', ())
    if any(type(layer).__name__ != 'DynamicLayer' for layer in layers):
        return None
    return ('DynamicCache', tuple(type(layer) for layer in layers), getattr(cache, 'layer_class_to_replicate', None))


class PrefillReplay:
    """Context-local single-entry cache with aggregate hit/miss/GPU timing."""
    def __init__(self, model):
        self.model = model
        self.original = model.forward
        self.signature = inspect.signature(self.original)
        self.key = None
        self.payload = None
        self.events = []
        self.counts = {
            'prefill_misses': 0, 'prefill_hits': 0, 'uncached_forwards': 0,
            'unsupported_forwards': 0, 'prefill_forward_wall_seconds': 0.,
            'replay_wall_seconds': 0., 'snapshot_wall_seconds': 0.,
            'prefill_gpu_seconds': 0., 'replay_gpu_seconds': 0.,
            'cache_capacity_prompts': 1,
        }

    def _key(self, args, kwargs):
        values = dict(self.signature.bind_partial(*args, **kwargs).arguments)
        ids = values.get('input_ids')
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] <= 1:
            return None, None
        if values.get('inputs_embeds') is not None or values.get('use_cache') is False:
            return None, None
        if values.get('output_attentions') or values.get('output_hidden_states') or values.get('return_dict') is False:
            return None, None
        cache = values.pop('past_key_values', None)
        signature = _empty_cache_signature(cache)
        if signature is None:
            return None, None
        # Keep every other forward argument (including masks, positions, and
        # logits_to_keep) in the exact comparison; no digest or partial matching.
        values['__empty_cache_signature__'] = signature
        return values, ids.device

    @staticmethod
    def _event(device):
        if device.type != 'cuda':
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(device))
        return event

    def _timed_end(self, begin, device, kind):
        if begin is not None:
            end = self._event(device)
            self.events.append((kind, begin, end))

    def forward(self, *args, **kwargs):
        try:
            key, device = self._key(args, kwargs)
        except (TypeError, ValueError):
            key, device = None, None
            self.counts['unsupported_forwards'] += 1
        if key is None:
            self.counts['uncached_forwards'] += 1
            return self.original(*args, **kwargs)
        if self.key is not None and _same(key, self.key):
            start = time.perf_counter()
            begin = self._event(device)
            # Deepcopy preserves DynamicCache metadata and storage independence.
            # Generate mutates the returned cache; the retained payload never escapes.
            output = copy.deepcopy(self.payload)
            self._timed_end(begin, device, 'replay_gpu_seconds')
            self.counts['prefill_hits'] += 1
            self.counts['replay_wall_seconds'] += time.perf_counter() - start
            return output
        try:
            saved_key = _snapshot(key)
        except TypeError:
            self.counts['unsupported_forwards'] += 1
            return self.original(*args, **kwargs)
        self.key = None
        self.payload = None
        start = time.perf_counter()
        begin = self._event(device)
        output = self.original(*args, **kwargs)
        self._timed_end(begin, device, 'prefill_gpu_seconds')
        self.counts['prefill_misses'] += 1
        self.counts['prefill_forward_wall_seconds'] += time.perf_counter() - start
        cache = getattr(output, 'past_key_values', None)
        if type(cache) is DynamicCache and getattr(output, 'logits', None) is not None:
            start = time.perf_counter()
            self.payload = copy.deepcopy(output)
            self.key = saved_key
            self.counts['snapshot_wall_seconds'] += time.perf_counter() - start
        return output

    def stats(self):
        # Event completion is resolved only on explicit statistics collection, not
        # on every decode token. CPU timing above is explicitly dispatch wall time.
        events, self.events = self.events, []
        for kind, begin, end in events:
            end.synchronize()
            self.counts[kind] += begin.elapsed_time(end) / 1000.
        return dict(self.counts)

    def __enter__(self):
        @functools.wraps(self.original)
        def wrapped(*args, **kwargs):
            return self.forward(*args, **kwargs)
        self.wrapped = wrapped
        self.model.forward = wrapped
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.model.forward = self.original
        self.key = None
        self.payload = None
        return False


@contextmanager
def replay_prefill(model):
    """Yield PrefillReplay for one complete score (path and outcome repeats)."""
    with PrefillReplay(model) as replay:
        yield replay
