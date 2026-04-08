# rinzler-gpt

A GPT language model trained end-to-end in Ruby. Transformer decoder with multi-head causal self-attention, trained via backpropagation through `rinzler-tensor`.

## Architecture

| Hyperparameter | Default |
|----------------|---------|
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `context_len` | 128 |
| `ffn_mult` | 4 (FFN hidden = 4 × d_model) |
| `vocab_size` | 1250 BPE merges |

Each transformer block: LayerNorm → multi-head attention → residual → LayerNorm → FFN (GELU) → residual. Output: linear projection to vocab logits → cross-entropy loss.

## Training

```bash
# Basic run
bundle exec ruby train.rb

# Full options
bundle exec ruby train.rb \
  --steps 100000 \
  --lr 3e-4 \
  --batch-size 8 \
  --context 128 \
  --d-model 64 \
  --layers 4 \
  --vocab-size 1250 \
  --warmup-steps 500 \
  --cosine \
  --clip-grad 1.0 \
  --save-every 500

# Resume from checkpoint
bundle exec ruby train.rb --resume rinzler-gpt/runs/model_step_5000.json

# Use a Rails app as corpus (filters .js/.jsx/.ts/.tsx/.css etc.)
bundle exec ruby train.rb --rails /path/to/rails/app

# Force retrain tokenizer (otherwise loads cached .json)
bundle exec ruby train.rb --retrain-tokenizer
```

## Corpus

Default: `corpus/*.txt`. `--corpus GLOB` or `--rails PATH` to add more sources. Multiple flags accumulate.

`--rails` crawls `app/`, `lib/`, `config/`, `db/` and filters out JS/React/CSS files so the model sees only Ruby. Filtered extensions: `.js .jsx .ts .tsx .mjs .cjs .css .scss .sass .less .svg`.

## Startup caching

The tokenizer vocabulary is cached to `tokenizer_<vocab_size>.json`. Encoded token IDs for the full corpus are cached to `tokenizer_<vocab_size>_ids.bin` (packed little-endian int32). On subsequent runs, the binary cache is loaded directly — skipping the O(corpus_size × vocab_size) encoding pass. Use `--retrain-tokenizer` to rebuild both.

## Divergence detection

The trainer monitors the train/val gap as a fraction of val loss. Two thresholds:

| Flag | Default | Behavior |
|------|---------|----------|
| `--div-warn N` | 20% | prints a warning |
| `--div-crit N` | 50% | aborts training |

`--div-window N` controls how many eval intervals to look back when computing the trend (default: 5).

## Profiling

```bash
bundle exec ruby rinzler-gpt/profile_step.rb
```

Two-phase output:

1. **Stage breakdown** — wall-clock per step for `zero_grad`, forward, backward, `free_graph!`, optimizer step.
2. **Backward op breakdown** — time per op type across the graph, showing which ops dominate the backward pass.

Note: `stackprof` and `ruby-prof` both segfault on Ruby 4.x (`METHOD_ENTRY_CACHED_SET` in `vm_insnhelper.c`). The profiler uses manual `Process.clock_gettime` timing instead.

## Known performance profile (batch=8, context=128, d_model=64)

At this scale, all matmuls are below the OpenBLAS threshold (256). The backward pass dominates at ~57% of step time.

Approximate step time distribution:

| stage | % of step |
|-------|-----------|
| backward | ~58% |
| forward | ~30% |
| opt_step | ~8% |
| zero_grad + free_graph | ~4% |

See `rinzler-tensor/README.md` for the per-op backward breakdown.

## Future directions

- **Larger d_model** — increasing from 64 to 128+ would push QKV projections and FFN matmuls above the BLAS threshold, routing them through OpenBLAS and giving a significant compute speedup.
- **Gradient checkpointing** — at larger models, the forward activations retained for backward dominate memory. Recomputing activations during backward trades memory for compute.
- **Multi-GPU / Vulkan backend** — `rinzler-tensor` supports routing `dot`/`bmm` through Vulkan compute shaders (`require "rinzler/vulkan"`, `--vulkan` flag). Not yet validated at training scale.
- **BPE vocab scaling** — corpus experiments show 1250 merges is a reasonable trade-off between step time and sequence compression. 2000+ merges slows training steps noticeably at d_model=64; revisit with larger d_model.
