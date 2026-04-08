# rinzler-optim

Gradient-based optimizers and LR schedulers for Rinzler. All optimizers share a common base interface; schedulers wrap any optimizer and adjust its learning rate each step.

## Optimizers

All optimizers take `parameters` (array of `Parameter` tensors) and `lr:`.

```ruby
opt = Rinzler::Optim::AdamW.new(model.parameters, lr: 3e-4, weight_decay: 0.1)

# Training loop
steps.times do
  opt.zero_grad
  loss.backward
  opt.step
end
```

| Class | Description |
|-------|-------------|
| `SGD` | Vanilla stochastic gradient descent |
| `SGDMomentum` | SGD with momentum (`momentum:` kwarg) |
| `RMSprop` | Adaptive per-parameter learning rates via running squared-gradient average |
| `Adam` | Momentum + adaptive rates + bias correction (Kingma & Ba, 2014) |
| `AdamW` | Adam with decoupled weight decay (Loshchilov & Hutter, 2017) — recommended for transformers |

### Gradient clipping

All optimizers expose `clip_grad_norm!(max_norm)`. Call it after `.backward`, before `.step`:

```ruby
opt.zero_grad
loss.backward
opt.clip_grad_norm!(1.0)   # returns pre-clip global norm
opt.step
```

### Checkpoint state

AdamW saves and restores moment buffers so training resumes exactly:

```ruby
state = opt.checkpoint_state          # → hash with m, v, step_count
opt.load_checkpoint_state!(state)
```

## Schedulers

Schedulers wrap an optimizer and expose the same interface. Swap `opt.step` → `scheduler.step`.

```ruby
opt       = Rinzler::Optim::AdamW.new(model.parameters, lr: 3e-4, weight_decay: 0.1)
scheduler = Rinzler::Optim::LinearWarmup.new(opt, warmup_steps: 500)

steps.times do
  scheduler.zero_grad
  loss.backward
  opt.clip_grad_norm!(1.0)
  scheduler.step
end
```

| Class | Description |
|-------|-------------|
| `LinearWarmup` | Ramp lr from 0 → base over `warmup_steps`, then hold |
| `CosineWithWarmup` | Linear warmup then cosine decay to `min_lr` (default 0) |

```ruby
# Cosine with warmup
scheduler = Rinzler::Optim::CosineWithWarmup.new(
  opt,
  warmup_steps: 500,
  total_steps:  50_000,
  min_lr:       1e-5
)
```
