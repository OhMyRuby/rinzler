# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/rinzler/autograd"

include Rinzler::Autograd

class TestValue < Minitest::Test
  EPSILON = 1e-6

  def assert_close(expected, actual, msg = nil)
    assert_in_delta expected, actual, EPSILON, msg
  end

  # ── Forward pass ────────────────────────────────────────────────────────────

  def test_addition
    a = Value.new(2.0)
    b = Value.new(3.0)
    assert_close 5.0, (a + b).data
  end

  def test_multiplication
    a = Value.new(2.0)
    b = Value.new(3.0)
    assert_close 6.0, (a * b).data
  end

  def test_power
    a = Value.new(3.0)
    assert_close 9.0, (a**2).data
  end

  def test_negation
    a = Value.new(4.0)
    assert_close(-4.0, (-a).data)
  end

  def test_subtraction
    a = Value.new(5.0)
    b = Value.new(2.0)
    assert_close 3.0, (a - b).data
  end

  def test_division
    a = Value.new(6.0)
    b = Value.new(2.0)
    assert_close 3.0, (a / b).data
  end

  def test_tanh
    a = Value.new(0.0)
    assert_close 0.0, a.tanh.data
  end

  def test_relu_passes_positive_through
    a = Value.new(3.0)
    assert_close 3.0, a.relu.data
  end

  def test_relu_clamps_negative_to_zero
    a = Value.new(-3.0)
    assert_close 0.0, a.relu.data
  end

  def test_exp
    a = Value.new(1.0)
    assert_close Math::E, a.exp.data
  end

  def test_log
    a = Value.new(Math::E)
    assert_close 1.0, a.log.data
  end

  def test_coerce_numeric_left_addition
    a = Value.new(3.0)
    assert_close 5.0, (2 + a).data
  end

  def test_coerce_numeric_left_multiplication
    a = Value.new(3.0)
    assert_close 6.0, (2 * a).data
  end

  # ── Backward pass ───────────────────────────────────────────────────────────
  # We verify gradients numerically using finite differences:
  # grad ≈ (f(x + h) - f(x - h)) / 2h
  # If our analytical gradient matches this, backprop is correct.

  def numerical_grad(val, expr, h: 1e-5)
    val.data += h
    fph = expr.call.data
    val.data -= 2 * h
    fmh = expr.call.data
    val.data += h
    (fph - fmh) / (2 * h)
  end

  def test_gradient_of_addition
    a = Value.new(2.0)
    b = Value.new(3.0)
    out = a + b
    out.backward
    assert_close 1.0, a.grad
    assert_close 1.0, b.grad
  end

  def test_gradient_of_multiplication
    a = Value.new(2.0)
    b = Value.new(3.0)
    out = a * b
    out.backward
    # d(a*b)/da = b = 3.0
    assert_close 3.0, a.grad
    # d(a*b)/db = a = 2.0
    assert_close 2.0, b.grad
  end

  def test_gradient_of_power
    a = Value.new(3.0)
    out = a**3
    out.backward
    # d(a^3)/da = 3a^2 = 27
    assert_close 27.0, a.grad
  end

  def test_gradient_of_tanh_matches_numerical
    a = Value.new(0.8)
    a.tanh.backward
    ng = numerical_grad(a, -> { Value.new(a.data).tanh })
    assert_close ng, a.grad
  end

  def test_gradient_of_relu_positive
    a = Value.new(2.0)
    a.relu.backward
    assert_close 1.0, a.grad
  end

  def test_gradient_of_relu_negative
    a = Value.new(-2.0)
    a.relu.backward
    assert_close 0.0, a.grad
  end

  def test_gradient_of_exp_matches_numerical
    a = Value.new(1.5)
    a.exp.backward
    ng = numerical_grad(a, -> { Value.new(a.data).exp })
    assert_close ng, a.grad
  end

  def test_gradient_of_log_matches_numerical
    a = Value.new(2.0)
    a.log.backward
    ng = numerical_grad(a, -> { Value.new(a.data).log })
    assert_close ng, a.grad
  end

  def test_gradients_through_longer_chain
    # loss = ((a * b) + c) ** 2
    a = Value.new(1.0)
    b = Value.new(2.0)
    c = Value.new(3.0)
    loss = ((a * b) + c)**2
    loss.backward

    ng_a = numerical_grad(a, -> { ((Value.new(a.data) * Value.new(b.data)) + Value.new(c.data))**2 })
    ng_b = numerical_grad(b, -> { ((Value.new(a.data) * Value.new(b.data)) + Value.new(c.data))**2 })
    ng_c = numerical_grad(c, -> { ((Value.new(a.data) * Value.new(b.data)) + Value.new(c.data))**2 })

    assert_close ng_a, a.grad, "gradient of a"
    assert_close ng_b, b.grad, "gradient of b"
    assert_close ng_c, c.grad, "gradient of c"
  end
end
