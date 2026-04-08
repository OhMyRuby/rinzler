# rinzler-nn

Neural network primitives built on `rinzler-tensor`. Provides the composable building blocks used to construct the GPT model.

## Layers

### `Linear`

Learned affine transformation: `y = xW + b`. Supports 2D `[batch, in_features]` and 3D `[batch, seq, in_features]` inputs — flattens leading dimensions automatically.

```ruby
layer = Rinzler::NN::Linear.new(64, 256)             # bias: true by default
layer = Rinzler::NN::Linear.new(64, 256, bias: false)
y = layer.call(x)
```

### `LayerNorm`

Pre-LN normalization. Normalizes along the last axis, then applies learned `gamma` (scale) and `beta` (shift). Numerically stable. 3D-aware.

```ruby
ln = Rinzler::NN::LayerNorm.new(64)
y  = ln.call(x)
```

### `Embedding`

Learned lookup table mapping integer token IDs to dense vectors. Handles batched index input; gradients accumulate correctly for repeated indices across a batch.

```ruby
emb = Rinzler::NN::Embedding.new(vocab_size, d_model)
y   = emb.call([[1, 4, 7], [2, 5, 8]])   # [B, T] → [B, T, d_model]
```

### `Module`

Base class for composable neural network components. Walks instance variables at call time to collect all `Parameter` tensors in the module and any nested modules.

```ruby
class MyModel < Rinzler::NN::Module
  def initialize
    @fc = Rinzler::NN::Linear.new(10, 10)
  end
  def forward(x) = @fc.call(x)
end

model = MyModel.new
model.parameters   # → flat array of all Parameter tensors
model.call(x)      # → delegates to forward
```

### `Parameter`

Thin wrapper that marks a `Tensor` as trainable. `Module#parameters` collects these recursively.
