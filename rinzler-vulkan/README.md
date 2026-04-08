# rinzler-vulkan

Optional GPU compute backend via Vulkan compute shaders. Routes `dot` and `bmm` through a tiled GLSL GEMM shader, leaving autograd on the CPU.

## What it does

The forward pass of a transformer is dominated by matrix multiplications — attention scores, value aggregation, and the vocabulary projection. On integrated AMD graphics, the Vulkan path crosses over at around n=512. The vocabulary projection `[B×T, d_model] × [d_model, vocab_size]` is the primary beneficiary during training.

Backward pass gradients remain on CPU. At small model scale this is the right trade-off: GPU→CPU transfers for grad computation are expensive, and the gradient matrices are smaller than the forward matmuls.

## Requirements

- `vulkan-headers`
- `vulkan-icd-loader` or `vulkan-radeon`
- `shaderc` (`glslc`) — for compiling the GLSL shader to SPIR-V at build time

## Build

```bash
cd rinzler-vulkan
bundle install
bundle exec rake compile
```

The SPIR-V binary is embedded in the gem at build time.

## Usage

```ruby
require "rinzler/tensor"
require "rinzler/vulkan"

Rinzler::Tensor.backend = :vulkan
# All subsequent dot/bmm calls route through GPU
```

Or via `train.rb`:

```bash
bundle exec ruby train.rb --vulkan ...
```

## Limitations

- Single-GPU only, no multi-device
- Data serializes through Ruby arrays on each call — no zero-copy path
- Vulkan initialization adds ~100ms startup overhead
