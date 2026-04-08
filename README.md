# Rinzler

A from-scratch GPT implementation in Ruby. The goal is to make every layer of a modern language model legible — not just usable, but *understandable* — by building it up from first principles, in historical order, without hiding the math behind framework magic.

This is not a production ML system. It is a pedagogical one. The code tells the story of how neural networks work.

## Philosophy

The codebase is organized as a dependency ladder. Each gem is complete and useful on its own, and each one builds on the last. You can stop at any rung and have a working, comprehensible system.

```
rinzler-autograd   ← scalar autograd (understand backprop on one number)
       ↓
rinzler-tensor     ← matrix autograd (make it practical)
       ↓
rinzler-nn         ← building blocks (Linear, LayerNorm, Embedding)
       ↓
rinzler-optim      ← AdamW optimizer
       ↓
rinzler-tokenizer  ← BPE tokenizer
       ↓
rinzler-gpt        ← GPT model + training loop
       ↓
rinzler-vulkan     ← GPU compute backend (optional)
```

## Modules

### `rinzler-autograd`
Scalar reverse-mode automatic differentiation. A `Value` wraps a single float and records every operation performed on it. Calling `.backward` on the loss walks the computation graph in reverse and accumulates gradients at every node via the chain rule.

This is the Karpathy micrograd approach — implement it once at the scalar level and you understand backprop completely.

### `rinzler-tensor`
The same autograd idea applied to Numo::DFloat arrays. Instead of tracking individual numbers, every operation (add, mul, dot, bmm, softmax, log\_softmax, layer\_norm, etc.) records how to backpropagate through a whole matrix at once. This is what makes training practical.

Supports a selectable backend:

```ruby
Rinzler::Tensor.backend = :vulkan  # route dot/bmm through GPU
Rinzler::Tensor.backend = :cpu     # default
```

Key ops: `dot`, `bmm`, `transpose_last2`, `softmax`, `log_softmax`, `sum`, `mean`, `reshape`, `slice_cols`, `concat_cols`.

### `rinzler-nn`
Neural network primitives built on top of `rinzler-tensor`:

- **`Linear`** — learned weight matrix + optional bias; supports 2D and 3D inputs (flattens leading dims automatically)
- **`LayerNorm`** — Pre-LN normalization; numerically stable; 3D-aware
- **`Embedding`** — learned lookup table; handles batched index input; gradient accumulates correctly for repeated indices
- **`Parameter`** — thin wrapper that marks a tensor as trainable

### `rinzler-optim`
Optimizers and LR schedulers.

Optimizers: `SGD`, `SGDMomentum`, `RMSprop`, `Adam`, `AdamW`. All support `clip_grad_norm!(max_norm)` for gradient clipping. AdamW supports checkpoint save/load of moment state so training resumes exactly.

Schedulers wrap any optimizer and adjust `lr` each step: `LinearWarmup` (ramp from 0 → base over N steps), `CosineWithWarmup` (warmup then cosine decay). Both expose the same interface as an optimizer.

### `rinzler-tokenizer`
Byte-Pair Encoding tokenizer. Character-stream approach (no word splitting) — tokens naturally include spaces, which is correct for Ruby code and prose where whitespace is semantic. `decode` is just `tokens.join`.

Trained in-session from the corpus; the vocabulary is saved alongside model checkpoints.

### `rinzler-gpt`
GPT-2 style transformer:

- Pre-LayerNorm architecture
- Multi-head causal self-attention with learned position embeddings (causal mask precomputed once at construction)
- GELU activation in FFN layers (tanh approximation, as used in GPT-2)
- Configurable depth/width via `Config`
- Vectorized cross-entropy loss through `log_softmax` (full gradient path preserved)
- Binary checkpoint format: weights and optimizer moments as raw float bytes + JSON sidecar for metadata. Legacy JSON-only format still loads.
- Graceful shutdown: `SIGINT`/`SIGTERM` sets a flag, training finishes the current step and saves a checkpoint before exiting.

**All scripts run from the monorepo root** so Bundler resolves inter-gem dependencies correctly:

```bash
# Find the fastest batch_size × OMP_NUM_THREADS combo before committing to a long run
bundle exec ruby rinzler-gpt/autotune.rb --vulkan

# Fresh run
bundle exec ruby rinzler-gpt/train.rb \
  --corpus "rinzler-gpt/corpus/*.txt" \
  --steps 50000 \
  --vocab-size 1000 \
  --warmup-steps 500 \
  --cosine \
  --div-crit 50 \
  --gen-every 500 \
  --vulkan

# Resume
bundle exec ruby rinzler-gpt/train.rb \
  --corpus "rinzler-gpt/corpus/*.txt" \
  --steps 100000 \
  --vocab-size 1000 \
  --warmup-steps 500 \
  --cosine \
  --div-crit 50 \
  --resume rinzler-gpt/runs/4/checkpoint_step15000.json \
  --vulkan
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--steps N` | 1000 | Training steps |
| `--lr N` | 3e-4 | Learning rate |
| `--batch-size N` | 8 | Sequences per step |
| `--context N` | 128 | Max sequence length |
| `--d-model N` | 64 | Embedding dimension |
| `--layers N` | 4 | Transformer blocks |
| `--vocab-size N` | 500 | BPE merge count |
| `--warmup-steps N` | 0 | LR warmup steps |
| `--cosine` | off | Cosine decay after warmup (requires `--warmup-steps`) |
| `--clip-grad N` | 1.0 | Gradient clipping max norm |
| `--no-clip-grad` | — | Disable gradient clipping |
| `--gen-every N` | 200 | Steps between text samples |
| `--save-every N` | 500 | Steps between checkpoints |
| `--div-warn N` | 20 | Warn when train/val gap exceeds N% |
| `--div-crit N` | — | Stop when train/val gap exceeds N% |
| `--vulkan` | off | Use GPU backend |
| `--resume PATH` | — | Resume from checkpoint (.json or .bin) |
| `--corpus PATTERN` | corpus.txt | Repeatable glob |

**Autotune** (`rinzler-gpt/autotune.rb`):

Benchmarks `batch_size` × `OMP_NUM_THREADS` combinations over a short fixed run, measures tokens/sec, and emits the optimal `train.rb` invocation. Accepts `--batch-sizes "4,8,16,32"`, `--threads "1,2,4"`, `--steps N`, `--vulkan`.

**Corpus** (`rinzler-gpt/corpus/`):
- `corpus.txt` — _why's Poignant Guide to Ruby
- `learn-to-program.txt` — Chris Pine's Learn to Program
- `pickaxe6_clean.txt` — Programming Ruby 4 (cleaned from PDF)

### `rinzler-vulkan`
Optional GPU compute backend via Vulkan compute shaders. Implements tiled 16×16 GEMM on the GPU using a GLSL compute shader compiled to SPIR-V at gem build time.

Crossover point on integrated AMD graphics: ~n=512. The vocabulary projection layer (`[B×T, d_model] × [d_model, vocab_size]`) is the primary beneficiary during training. Large inference batches benefit most.

**Requirements:** `vulkan-headers`, `vulkan-icd-loader` (or `vulkan-radeon`), `shaderc` (for `glslc`).

**Build:**
```bash
cd rinzler-vulkan
bundle install
bundle exec rake compile
```

## Current State

### What works
- Full training loop: forward pass → loss → backward pass → AdamW step
- Batched training (multiple sequences per step)
- BPE tokenization with 1000-merge vocabulary
- Binary checkpoint format (raw floats + JSON sidecar) with backwards-compatible JSON-only loader
- Checkpoint save/resume: model weights + AdamW moment state + tokenizer
- Multi-corpus support (glob patterns, merged at load time)
- GPU acceleration via Vulkan (dot + bmm backend)
- Text generation with temperature sampling
- Linear LR warmup (`LinearWarmup`, `CosineWithWarmup` schedulers)
- Gradient clipping (`clip_grad_norm!`)
- GELU activation (GPT-2 tanh approximation)
- Causal mask precomputed at model construction
- Graceful `SIGINT`/`SIGTERM` shutdown with checkpoint save
- Divergence monitor: configurable warn/stop thresholds on train/val gap

### What's been trained
Run 4 is in progress: 1000-merge BPE vocabulary, full corpus (_why + Chris Pine + Pickaxe), step ~15k of 50k. Generation at step 13.5k shows correct Ruby token patterns and rough prose structure.

### Known limitations
- No KV cache — generation is O(T²) per token
- Single-GPU only, no multi-device
- Vulkan path serializes data through Ruby arrays (no zero-copy)
- Model is small by current standards (64d); coherent generation but limited range

### Deferred
- KV cache for O(T) generation (currently O(T²) per token)
- `CosineWithWarmup` resume support: `total_steps` should account for `start_step` on resume

## Dependencies

- Ruby 4.0+
- [numo-narray](https://github.com/ruby-numo/numo-narray) (local fork at `../numo-narray`)
- Vulkan SDK (optional, for `rinzler-vulkan`)
