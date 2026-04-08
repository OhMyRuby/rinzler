# rinzler-tensor

Matrix reverse-mode automatic differentiation. The same idea as `rinzler-autograd` but operating on entire Numo::DFloat arrays at once.

## What it does

`Tensor` wraps a Numo::DFloat n-dimensional array and tracks the computation graph for backpropagation. Every operation records how to distribute gradients back through it. Calling `.backward` on the loss walks the graph and accumulates gradients at every parameter.

This is why neural networks are practical: instead of backprop through millions of individual scalars, you run it through a handful of matrix operations in batch.

## Backends

```ruby
# Default: CPU (OpenBLAS via numo-linalg for matmuls >= 256d)
Rinzler::Tensor.backend = :cpu

# GPU: route dot/bmm through Vulkan compute shaders
require "rinzler/vulkan"
Rinzler::Tensor.backend = :vulkan
```

## Operations

```ruby
T = Rinzler::Tensor::Tensor

a = T.randn(4, 64)
b = T.randn(64, 32)

# Arithmetic (broadcast-aware)
a + b;  a * b;  a - b;  a / b;  a ** 2

# Matrix multiply
a.dot(b)              # [4, 64] × [64, 32] → [4, 32]
a.bmm(b)              # batched: [B, T, d] × [B, d, S] → [B, T, S]

# Reductions
x.sum                 # scalar
x.sum(axis: 1)        # reduce along axis
x.mean

# Shape
x.reshape(2, 128)
x.transpose           # 2D only
x.transpose_last2     # last two dims — used in attention

# Slicing (differentiable)
x.slice_cols(0, 64)
T.concat_cols([a, b])

# Activations
x.relu
x.tanh
x.exp
x.log
x.softmax
x.log_softmax         # numerically stable; used in cross-entropy loss
```

## Factory methods

```ruby
T.zeros(4, 64)
T.ones(4, 64)
T.randn(4, 64)
T.rand(4, 64)
T.from([[1.0, 2.0], [3.0, 4.0]])
```

## Native extension (tensor_ext)

`bmm` routes through a C extension (`tensor_ext`) that loops over batch items in C rather than Ruby, accessing numo NArray data pointers directly. With `-O3 -march=native` the inner k-loop vectorises with AVX2.

Build: `bundle exec rake compile` inside `rinzler-tensor/`.

The extension falls back gracefully if not compiled — the Ruby path is preserved.

**Load order matters:** `tensor_ext.so` resolves `na_data_type` from the already-loaded `narray.so` at runtime. `tensor.rb` requires `numo/narray` (via `tensor/tensor.rb`) before loading the extension. Do not change this order.

## Backward pass design

All gradient accumulations use numo's in-place operations (`grad.inplace + delta`) to avoid allocating a new NArray per backward node. Profiling at training scale (batch=8, context=128, d_model=64) showed this cut backward time by ~28% compared to `grad = grad + delta`.

The `sum`/`mean` backward passes broadcast `out.grad` directly rather than allocating a `ones` array and multiplying — numo handles the broadcast.

## Known performance profile (training scale: batch=8, context=128, d_model=64)

Approximate backward time distribution per step:

| op | % of backward |
|----|---------------|
| `dot` | ~27% |
| `bmm` | ~23% |
| `*` | ~15% |
| `log_softmax` | ~10% |
| `reshape` | ~9% |

`dot` and `bmm` are memory-bound compute — further gains require fused ops. `log_softmax` operates on the full `[B*T, vocab_size]` logit matrix and dominates single-node cost.

`*` backward uses `TensorExt.mul_backward` when both operands have matching shapes — a single C loop accumulates into both `self.grad` and `other.grad` with no temporaries, halving memory traffic vs. the two-pass numo path. Mismatched shapes (scalar broadcast etc.) fall back to Ruby.

## Fused cross-entropy loss

`Tensor.cross_entropy(logits, targets)` replaces the naive `log_softmax → (* one_hot) → sum → mean` chain. That chain allocates a `[B*T, vocab_size]` one_hot matrix (~10 MB at training scale) and runs 5 separate backward nodes. The fused version:

- Forward: numerically stable log-softmax then integer-indexed gather — no one_hot alloc
- Backward: single node computing `(softmax(logits) - one_hot) / n` directly, with the target-position subtract done via Ruby iteration on the 2D grad array (O(B*T), no auxiliary allocation)

Note: numo `reshape` returns a copy, not a view, so flat-index tricks don't work for in-place grad modification — the 2D iteration is the correct approach.

## Gradient buffer allocation

All backward closures use `self.grad.inplace + delta` — no backward path reassigns `@grad` to a new NArray. Combined with the optimizer's `zero_grad` using `grad.fill(0.0)`, parameter tensors' grad NArrays are allocated exactly once (at construction) and reused for the entire training run. No GC pressure on parameter grads across steps.

## Future optimization paths

- **Larger d_model** — most matmuls are below the BLAS threshold (256) at d_model=64. Increasing d_model to 128+ would route the QKV projection and FFN layers through OpenBLAS, giving a significant speedup on the dominant ops.
