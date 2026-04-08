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
a.bmm(b)              # batched: [B, T, d] × [B, d, T] → [B, T, T]

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
