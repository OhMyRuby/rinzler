# rinzler-autograd

Scalar reverse-mode automatic differentiation. The first rung of the Rinzler dependency ladder.

## What it does

`Value` wraps a single float and records every arithmetic operation performed on it. Calling `.backward` on the final output walks the computation graph in reverse topological order and accumulates gradients at every node via the chain rule.

This is the Karpathy micrograd approach: implement backprop once at the scalar level, and you understand it completely before scaling up to matrices.

## Usage

```ruby
require "rinzler/autograd"

a = Rinzler::Autograd::Value.new(2.0)
b = Rinzler::Autograd::Value.new(3.0)

loss = (a * b + b) ** 2
loss.backward

a.grad  # d(loss)/da
b.grad  # d(loss)/db
```

## Supported operations

| Operation | Backward rule |
|-----------|---------------|
| `+`       | Gradient passes through unchanged to both inputs |
| `*`       | Each input's gradient scaled by the other input's value |
| `**n`     | Power rule: `n * x^(n-1)` |
| `-`, `/`  | Defined in terms of `+`, `*`, `**` |
| `tanh`    | `1 - tanh(x)²` |
| `relu`    | 1 if input > 0, else 0 |
| `exp`     | `e^x` (its own derivative) |
| `log`     | `1/x` |

## Design notes

- Scalars only — no batching, no matrices. That's intentional: the math is legible at this level.
- `coerce` is implemented so `2 + value` and `2 * value` work naturally.
- The topological sort in `backward` ensures each node's `_backward` runs only after all downstream gradients have accumulated.
