"""Reuse a case's exact token prefix with the backbone's native dynamic cache."""
from types import MethodType
from time import perf_counter
import torch
from experiments.m1_diagnostics.engine import Context


def attach_case_prefix(engine, texts: dict[str, str]) -> None:
    """Attach prepare() for one case; preserve every prompt and absolute position.

    A new call releases the previous case's root cache. Only the common prefix
    remains cached; each node's suffix is recomputed in its own shallow cache
    copy, just as the canonical engine replays its final token independently.
    """
    if not hasattr(engine, "_case_original_prepare"):
        engine._case_original_prepare = engine.prepare
        engine._case_prefix_stats = dict(cases=0, shared_tokens=0, suffix_tokens=0)
    prompts = list(dict.fromkeys(texts.values()))
    if not prompts:
        engine.prepare = engine._case_original_prepare
        engine._case_root_cache = None
        engine._case_token_ids = {}
        return
    # Drop old case GPU references before allocating a new prefix.
    engine._case_root_cache = None
    engine._case_token_ids = {}
    token_ids = {prompt: engine.ids(prompt) for prompt in prompts}
    token_lists = [ids[0].tolist() for ids in token_ids.values()]
    common = min(len(x) - 1 for x in token_lists)
    reference = token_lists[0]
    for other in token_lists[1:]:
        common = next((i for i in range(common) if reference[i] != other[i]), common)
    common = (common // engine.prefill_chunk) * engine.prefill_chunk
    first_ids = next(iter(token_ids.values()))
    started = perf_counter()
    root = engine._prefill(first_ids, common)
    engine.counts["prefills"] += 1
    engine.counts["prefill_seconds"] += perf_counter() - started
    engine._case_root_cache = root
    engine._case_token_ids = token_ids
    engine._case_prefix_length = common
    engine._case_prefix_stats["cases"] += 1
    engine._case_prefix_stats["shared_tokens"] += common

    def prepare(self, prompt, task="binary"):
        ids = self._case_token_ids.get(prompt) if task == "binary" else None
        if ids is None:
            return self._case_original_prepare(prompt, task)
        started = perf_counter()
        cache = self._copy_cache(self._case_root_cache)
        stop = ids.shape[1] - 1
        with torch.no_grad():
            for start in range(self._case_prefix_length, stop, self.prefill_chunk):
                end = min(start + self.prefill_chunk, stop)
                positions = torch.arange(start, end, device=self.device)
                output = self.model.model(
                    input_ids=ids[:, start:end],
                    attention_mask=torch.ones_like(ids[:, :end]),
                    past_key_values=cache, use_cache=True,
                    cache_position=positions,
                    position_ids=positions.unsqueeze(0))
                cache = output.past_key_values
                del output
        self._case_prefix_stats["suffix_tokens"] += stop - self._case_prefix_length
        self.counts["prefills"] += 1
        self.counts["prefill_seconds"] += perf_counter() - started
        return Context(prompt, ids, cache, int(ids.shape[1]))

    engine.prepare = MethodType(prepare, engine)


def validate_case_prefix(engine, texts: dict[str, str], output_path) -> dict:
    """Compare two actual prompts with normal and shared-prefix cache execution."""
    import json
    from pathlib import Path
    prompts = list(dict.fromkeys(texts.values()))[:2]
    original = getattr(engine, "_case_original_prepare", engine.prepare)
    direct = []
    for prompt in prompts:
        context = original(prompt)
        direct.append(engine.run(context, capture=False)["ell"])
        del context
    attach_case_prefix(engine, texts)
    shared = []
    for prompt in prompts:
        context = engine.prepare(prompt)
        shared.append(engine.run(context, capture=False)["ell"])
        del context
    result = {
        "protocol": "same real first-case prompts, canonical native cache vs shared exact token prefix",
        "model": engine.model_name,
        "shared_tokens": engine._case_prefix_length,
        "scores": [
            dict(direct=a, shared=b, absolute_difference=abs(a-b),
                 same_binary_prediction=(a > 0) == (b > 0))
            for a,b in zip(direct, shared)],
        "max_absolute_difference": max((abs(a-b) for a,b in zip(direct,shared)), default=0.0),
        "note": "Reuse only complete canonical prefill chunks; suffix boundaries match the original execution.",
    }
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    return result
