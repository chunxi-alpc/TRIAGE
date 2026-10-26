"""Method-independent repeated judge draws, with durable exact-request caching."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

from mega_development_evaluation import RequestRecorder, value_hash
from mega_local_qwen_client import LocalResponse


def restore_response(payload):
    return LocalResponse(payload=payload, id=payload.get('id'), model=payload.get('model'),
                         choices=[SimpleNamespace(message=SimpleNamespace(**v['message']),
                                                  finish_reason=v.get('finish_reason'))
                                  for v in payload['choices']],
                         usage=SimpleNamespace(**(payload.get('usage') or {})))


class RequestKeyedRecorder(RequestRecorder):
    """Cache five distinct ordinal draws; never collapse repeats into one draw.

    Each method/case/metric resets request ordinals. Method is excluded from cache
    keys and RNG seeds. Full request messages contain answers and reference text.
    Namespace includes frozen judge, model, reference and source fingerprints.
    Only actual requests reach the original cost recorder.
    """
    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = value
        self.ordinals = Counter()

    def __init__(self, client, raw_path, metadata_path, *, namespace, cache_dir, events_path):
        super().__init__(client, raw_path, metadata_path)
        self.namespace = namespace
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.events = Path(events_path).open('x')
        self.cache_hits = self.cache_misses = 0

    def create(self, **request):
        request_hash = value_hash(request)
        ordinal = self.ordinals[request_hash]
        self.ordinals[request_hash] += 1
        key = value_hash({'version': 1, 'namespace': self.namespace,
                          'sample_id': self.context['sample_id'],
                          'metric': self.context['metric'], 'request': request,
                          'draw_ordinal': ordinal})
        path = self.cache_dir / (key + '.json')
        seed = int(key[:16], 16) % (2**31)
        event = {**self.context, 'cache_key': key, 'request_sha256': request_hash,
                 'draw_ordinal': ordinal, 'request_seed': seed}
        try:
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['key'] != key or value_hash(saved['payload']) != saved['response_sha256']:
                    raise ValueError('Judge cache integrity mismatch')
                self.cache_hits += 1
                event.update(status='cache_hit', actual_forward_calls=0,
                             response_sha256=saved['response_sha256'])
                return restore_response(saved['payload'])
            self.cache_misses += 1
            self.client.torch.manual_seed(seed)
            self.client.torch.cuda.manual_seed_all(seed)
            response = super().create(**request)
            payload = response.model_dump()
            saved = {'key': key, 'request_seed': seed, 'payload': payload,
                     'response_sha256': value_hash(payload)}
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(saved, ensure_ascii=False) + '\n')
            temporary.replace(path)
            event.update(status='generated', response_sha256=saved['response_sha256'])
            return response
        except Exception as error:
            event.update(status='error', error_type=type(error).__name__)
            raise
        finally:
            self.events.write(json.dumps(event, ensure_ascii=False) + '\n')
            self.events.flush()

    def cost_summary(self):
        result = super().cost_summary()
        result['exact_request_cache'] = {'hits': self.cache_hits, 'misses': self.cache_misses,
                                        'hit_forward_cost': 0,
                                        'policy': 'method-independent keyed ordinal draws'}
        return result

    def close(self):
        super().close()
        self.events.close()
