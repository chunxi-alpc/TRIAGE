# Model change: Qwen3-4B-Base

Requested replacement: use Qwen/Qwen3-4B-Base for **all three preliminary observations**, rather than the previous 0.6B model.

Authoritative model configuration: https://huggingface.co/Qwen/Qwen3-4B-Base/raw/main/config.json

- Layers: 36
- Query attention heads: 32
- Key/value heads: 8
- Grouped-query ratio: 4 query heads per KV head
- Head dimension: 128
- BF16 weight bytes: approximately 8 GB (4B parameters × 2 bytes; additional memory required for runtime and model analyses)
- GPU recommendation: at least 24 GB for full pilot; actual memory depends on sequence length and implementation.
- Fallback model: disabled; do not silently replace the 4B model.
- The original SEKA/Pasta hooks and RoPE pre/post-rotation caveats are unchanged.
- This package does not include downloaded model weights or completed observation results.
