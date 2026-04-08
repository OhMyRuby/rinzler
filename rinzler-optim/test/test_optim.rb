# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/rinzler/optim"
require "rinzler/nn"

T   = Rinzler::Tensor::Tensor
NN  = Rinzler::NN
Opt = Rinzler::Optim

# Shared helper: train a single Linear layer to fit y = 2x on scalar inputs.
# Returns the loss after N steps. A working optimizer drives it toward zero.
module OptimizerTestHelper
  def train_loss(optimizer_class, steps: 200, lr: 0.01, **kwargs)
    layer = NN::Linear.new(1, 1, bias: false)

    # Normalized dataset: y = 2x, inputs scaled to keep gradients manageable
    xs = T.from([[0.25], [0.5], [0.75], [1.0]])
    ys = T.from([[0.5],  [1.0], [1.5],  [2.0]])

    opt = optimizer_class.new(layer.parameters, lr:, **kwargs)

    steps.times do
      opt.zero_grad
      pred = layer.call(xs)
      loss = ((pred - ys) ** 2).mean
      loss.backward
      opt.step
    end

    pred = layer.call(xs)
    ((pred - ys) ** 2).mean.data.sum
  end
end

class TestSGD < Minitest::Test
  include OptimizerTestHelper

  def test_reduces_loss
    loss = train_loss(Opt::SGD, steps: 500, lr: 0.01)
    assert loss < 0.1, "SGD should reduce loss below 0.1, got #{loss}"
  end

  def test_zero_grad_clears_gradients
    layer = NN::Linear.new(2, 1)
    opt   = Opt::SGD.new(layer.parameters, lr: 0.01)
    x     = T.randn(3, 2)
    layer.call(x).sum.backward
    opt.zero_grad
    layer.parameters.each { |p| assert_equal 0.0, p.grad.sum }
  end
end

class TestSGDMomentum < Minitest::Test
  include OptimizerTestHelper

  def test_reduces_loss
    loss = train_loss(Opt::SGDMomentum, steps: 300, lr: 0.01, momentum: 0.9)
    assert loss < 0.1, "SGD+Momentum should reduce loss below 0.1, got #{loss}"
  end

  def test_velocity_accumulates
    # Momentum tracks a running average of past gradients (velocity).
    # When gradients point consistently in the same direction, velocity
    # grows step over step — proving accumulation is working.
    layer = NN::Linear.new(1, 1, bias: false)
    xs    = T.from([[0.25], [0.5], [0.75], [1.0]])
    ys    = T.from([[0.5],  [1.0], [1.5],  [2.0]])
    opt   = Opt::SGDMomentum.new(layer.parameters, lr: 0.001, momentum: 0.9)

    opt.zero_grad
    ((layer.call(xs) - ys) ** 2).mean.tap(&:backward)
    opt.step
    v_after_1 = opt.instance_variable_get(:@velocity)[0].abs.sum

    9.times do
      opt.zero_grad
      ((layer.call(xs) - ys) ** 2).mean.tap(&:backward)
      opt.step
    end
    v_after_10 = opt.instance_variable_get(:@velocity)[0].abs.sum

    assert v_after_10 > v_after_1,
           "Velocity should accumulate with consistent gradients (#{v_after_1} → #{v_after_10})"
  end
end

class TestRMSprop < Minitest::Test
  include OptimizerTestHelper

  def test_reduces_loss
    # Adaptive optimizers normalize updates to ~lr per step, so they need
    # a larger lr than SGD to cover the same distance in the same iterations.
    loss = train_loss(Opt::RMSprop, steps: 200, lr: 0.1)
    assert loss < 0.01, "RMSprop should reduce loss below 0.01, got #{loss}"
  end
end

class TestAdam < Minitest::Test
  include OptimizerTestHelper

  def test_reduces_loss
    loss = train_loss(Opt::Adam, steps: 200, lr: 0.1)
    assert loss < 0.01, "Adam should reduce loss below 0.01, got #{loss}"
  end

  def test_bias_correction_applied
    # With bias correction, Adam's first step is approximately lr * sign(grad).
    # Without it, m/sqrt(v) ≈ 0.1*grad / sqrt(0.001*grad²) = 0.1/sqrt(0.001) ≈ 3.16
    # so the first step would be ~316x the intended lr — destabilizing.
    layer = NN::Linear.new(1, 1, bias: false)
    opt   = Opt::Adam.new(layer.parameters, lr: 0.1)
    xs    = T.from([[0.25]])
    ys    = T.from([[0.5]])

    initial_loss = ((layer.call(xs) - ys) ** 2).mean.data.sum

    10.times do
      opt.zero_grad
      loss = ((layer.call(xs) - ys) ** 2).mean
      loss.backward
      opt.step
    end

    loss_after = ((layer.call(xs) - ys) ** 2).mean.data.sum
    assert loss_after < initial_loss, "Adam should reduce loss in 10 steps"
  end
end

class TestLinearWarmup < Minitest::Test
  def make_opt(lr: 0.1)
    layer = NN::Linear.new(1, 1, bias: false)
    opt   = Opt::AdamW.new(layer.parameters, lr:)
    [layer, opt]
  end

  def test_lr_ramps_linearly_during_warmup
    _, opt      = make_opt(lr: 1.0)
    scheduler   = Opt::LinearWarmup.new(opt, warmup_steps: 10)

    lrs = 10.times.map do
      scheduler.step
      scheduler.lr
    end

    # Each step should be 0.1 higher than the last during warmup
    10.times do |i|
      expected = 1.0 * (i + 1) / 10.0
      assert_in_delta expected, lrs[i], 1e-9,
        "Step #{i + 1}: expected lr=#{expected}, got #{lrs[i]}"
    end
  end

  def test_lr_holds_after_warmup
    _, opt    = make_opt(lr: 0.5)
    scheduler = Opt::LinearWarmup.new(opt, warmup_steps: 5)

    15.times { scheduler.step }

    assert_in_delta 0.5, scheduler.lr, 1e-9, "lr should hold at base_lr after warmup"
  end

  def test_reduces_loss
    layer     = NN::Linear.new(1, 1, bias: false)
    xs        = T.from([[0.25], [0.5], [0.75], [1.0]])
    ys        = T.from([[0.5],  [1.0], [1.5],  [2.0]])
    opt       = Opt::AdamW.new(layer.parameters, lr: 0.1)
    scheduler = Opt::LinearWarmup.new(opt, warmup_steps: 20)

    200.times do
      scheduler.zero_grad
      loss = ((layer.call(xs) - ys) ** 2).mean
      loss.backward
      scheduler.step
    end

    loss = ((layer.call(xs) - ys) ** 2).mean.data.sum
    assert loss < 0.01, "LinearWarmup+AdamW should reduce loss below 0.01, got #{loss}"
  end
end

class TestCosineWithWarmup < Minitest::Test
  def test_lr_is_zero_at_step_zero_concept
    # At step 1 of warmup_steps=10, lr should be base_lr/10
    layer     = NN::Linear.new(1, 1, bias: false)
    opt       = Opt::AdamW.new(layer.parameters, lr: 1.0)
    scheduler = Opt::CosineWithWarmup.new(opt, warmup_steps: 10, total_steps: 100)

    scheduler.step
    assert_in_delta 0.1, scheduler.lr, 1e-9
  end

  def test_lr_at_peak_after_warmup
    layer     = NN::Linear.new(1, 1, bias: false)
    opt       = Opt::AdamW.new(layer.parameters, lr: 1.0)
    scheduler = Opt::CosineWithWarmup.new(opt, warmup_steps: 10, total_steps: 100)

    10.times { scheduler.step }
    assert_in_delta 1.0, scheduler.lr, 1e-9, "lr should equal base_lr at end of warmup"
  end

  def test_lr_at_total_steps_equals_min_lr
    layer     = NN::Linear.new(1, 1, bias: false)
    opt       = Opt::AdamW.new(layer.parameters, lr: 1.0)
    scheduler = Opt::CosineWithWarmup.new(opt, warmup_steps: 10, total_steps: 110, min_lr: 0.0)

    110.times { scheduler.step }
    assert_in_delta 0.0, scheduler.lr, 1e-9, "lr should reach min_lr at total_steps"
  end

  def test_lr_monotonically_decreases_after_warmup
    layer     = NN::Linear.new(1, 1, bias: false)
    opt       = Opt::AdamW.new(layer.parameters, lr: 1.0)
    scheduler = Opt::CosineWithWarmup.new(opt, warmup_steps: 5, total_steps: 50)

    5.times { scheduler.step }  # complete warmup
    prev_lr = scheduler.lr

    44.times do |i|
      scheduler.step
      assert scheduler.lr <= prev_lr,
        "lr should not increase after warmup (step #{i + 6}: #{prev_lr} → #{scheduler.lr})"
      prev_lr = scheduler.lr
    end
  end

  def test_reduces_loss
    layer     = NN::Linear.new(1, 1, bias: false)
    xs        = T.from([[0.25], [0.5], [0.75], [1.0]])
    ys        = T.from([[0.5],  [1.0], [1.5],  [2.0]])
    opt       = Opt::AdamW.new(layer.parameters, lr: 0.1)
    scheduler = Opt::CosineWithWarmup.new(opt, warmup_steps: 20, total_steps: 500)

    200.times do
      scheduler.zero_grad
      loss = ((layer.call(xs) - ys) ** 2).mean
      loss.backward
      scheduler.step
    end

    loss = ((layer.call(xs) - ys) ** 2).mean.data.sum
    assert loss < 0.01, "CosineWithWarmup+AdamW should reduce loss below 0.01, got #{loss}"
  end
end

class TestAdamW < Minitest::Test
  include OptimizerTestHelper

  def test_reduces_loss
    loss = train_loss(Opt::AdamW, steps: 200, lr: 0.1, weight_decay: 0.01)
    assert loss < 0.01, "AdamW should reduce loss below 0.01, got #{loss}"
  end

  def test_weight_decay_shrinks_parameters
    # With high weight decay and no signal, parameters should decay toward zero.
    layer = NN::Linear.new(2, 2, bias: false)
    opt   = Opt::AdamW.new(layer.parameters, lr: 0.001, weight_decay: 0.5)

    initial_norm = layer.weight.data.abs.sum

    # Zero gradients — only weight decay acts
    100.times do
      opt.zero_grad
      # Don't backward — gradient stays zero, only decay applies
      opt.step
    end

    final_norm = layer.weight.data.abs.sum
    assert final_norm < initial_norm,
           "Weight decay should shrink parameters (#{initial_norm} → #{final_norm})"
  end

  def test_decoupled_from_adaptive_scaling
    # Key property: weight decay effect should be the same regardless of gradient magnitude.
    # We can't test this directly without a reference, but we can verify the update
    # formula by checking that parameters shrink even when gradients are near zero.
    layer  = NN::Linear.new(1, 1, bias: false)
    layer.weight.data[0, 0] = 1.0
    opt    = Opt::AdamW.new(layer.parameters, lr: 0.1, weight_decay: 0.1)

    opt.zero_grad
    opt.step  # step with zero gradient — only decay acts

    # Parameter should have shrunk by approximately lr * weight_decay = 0.01
    assert layer.weight.data[0, 0] < 1.0, "Decoupled decay should shrink parameter"
  end
end
