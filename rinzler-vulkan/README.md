# rinzler-vulkan

Optional GPU compute backend via Vulkan compute shaders. Routes `dot` and `bmm` through a tiled GLSL GEMM shader, leaving autograd on the CPU.

## What it does

The forward pass of a transformer is dominated by matrix multiplications — attention scores, value aggregation, and the vocabulary projection. On integrated AMD graphics, the Vulkan path crosses over at around n=512.

**Important:** for small models (d_model ≤ 128), Vulkan is slower than CPU. Every matmul pays a Ruby→C→Vulkan→Ruby serialization cost that dominates for small matrices. Benchmarked on d_model=64: Vulkan is **2.4× slower** than OpenBLAS on CPU. Only enable Vulkan when d_model ≥ 256, or when the vocabulary projection `[B×T, d_model] × [d_model, vocab_size]` is large enough to amortize the overhead. Use `autotune.rb` with and without `--vulkan` to verify before a long run.

Backward pass gradients remain on CPU in all cases.

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
