# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/rinzler/tensor"

include Rinzler::Tensor

class TestTensor < Minitest::Test
  EPSILON = 1e-5
  DELTA   = 1e-3  # looser tolerance for numerical gradient comparison

  def assert_close(expected, actual, msg = nil)
    assert_in_delta expected, actual, DELTA, msg
  end

  # Numerical gradient for a scalar-output function of one tensor element.
  # Perturbs element [*idx] of tensor t, evaluates expr, returns finite-diff grad.
  def numerical_grad(t, idx, expr, h: 1e-4)
    t.data[*idx] += h
    fph = expr.call.data.sum
    t.data[*idx] -= 2 * h
    fmh = expr.call.data.sum
    t.data[*idx] += h
    (fph - fmh) / (2 * h)
  end

  # ── Construction ──────────────────────────────────────────────────────────────

  def test_from_array
    t = Tensor.from([[1.0, 2.0], [3.0, 4.0]])
    assert_equal [2, 2], t.shape
    assert_in_delta 1.0, t.data[0, 0], EPSILON
  end

  def test_zeros
    t = Tensor.zeros(3, 4)
    assert_equal [3, 4], t.shape
    assert_in_delta 0.0, t.data[0, 0], EPSILON
  end

  def test_ones
    t = Tensor.ones(2, 3)
    assert_equal [2, 3], t.shape
    assert_in_delta 1.0, t.data[1, 2], EPSILON
  end

  # ── Forward pass ──────────────────────────────────────────────────────────────

  def test_addition
    a = Tensor.from([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor.from([[5.0, 6.0], [7.0, 8.0]])
    c = a + b
    assert_in_delta 6.0,  c.data[0, 0], EPSILON
    assert_in_delta 12.0, c.data[1, 1], EPSILON
  end

  def test_multiplication
    a = Tensor.from([[2.0, 3.0]])
    b = Tensor.from([[4.0, 5.0]])
    c = a * b
    assert_in_delta 8.0,  c.data[0, 0], EPSILON
    assert_in_delta 15.0, c.data[0, 1], EPSILON
  end

  def test_dot
    a = Tensor.from([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor.from([[5.0, 6.0], [7.0, 8.0]])
    c = a.dot(b)
    # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    assert_in_delta 19.0, c.data[0, 0], EPSILON
    assert_in_delta 50.0, c.data[1, 1], EPSILON
  end

  def test_sum_all
    a = Tensor.from([[1.0, 2.0], [3.0, 4.0]])
    assert_in_delta 10.0, a.sum.data.sum, EPSILON
  end

  def test_mean_all
    a = Tensor.from([[1.0, 2.0], [3.0, 4.0]])
    assert_in_delta 2.5, a.mean.data.sum, EPSILON
  end

  def test_reshape
    a = Tensor.from([[1.0, 2.0], [3.0, 4.0]])
    b = a.reshape(1, 4)
    assert_equal [1, 4], b.shape
    assert_in_delta 3.0, b.data[0, 2], EPSILON
  end

  def test_transpose
    a = Tensor.from([[1.0, 2.0], [3.0, 4.0]])
    b = a.T
    assert_in_delta 3.0, b.data[0, 1], EPSILON
    assert_in_delta 2.0, b.data[1, 0], EPSILON
  end

  def test_relu_passes_positive
    a = Tensor.from([[1.0, -2.0], [-3.0, 4.0]])
    b = a.relu
    assert_in_delta 1.0, b.data[0, 0], EPSILON
    assert_in_delta 0.0, b.data[0, 1], EPSILON
    assert_in_delta 0.0, b.data[1, 0], EPSILON
    assert_in_delta 4.0, b.data[1, 1], EPSILON
  end

  def test_tanh_range
    a = Tensor.from([[0.0, 1.0]])
    b = a.tanh
    assert_in_delta 0.0,      b.data[0, 0], EPSILON
    assert_in_delta 0.761594, b.data[0, 1], EPSILON
  end

  def test_softmax_sums_to_one
    a = Tensor.from([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    s = a.softmax
    assert_in_delta 1.0, s.data[0, true].sum, EPSILON
    assert_in_delta 1.0, s.data[1, true].sum, EPSILON
  end

  # ── Backward pass ─────────────────────────────────────────────────────────────

  def test_gradient_of_addition
    a = Tensor.from([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor.from([[5.0, 6.0], [7.0, 8.0]])
    out = (a + b).sum
    out.backward

    # d(sum(a+b))/da = ones everywhere
    assert_in_delta 1.0, a.grad[0, 0], EPSILON
    assert_in_delta 1.0, a.grad[1, 1], EPSILON
    assert_in_delta 1.0, b.grad[0, 0], EPSILON
  end

  def test_gradient_of_multiplication
    a = Tensor.from([[2.0, 3.0]])
    b = Tensor.from([[4.0, 5.0]])
    out = (a * b).sum
    out.backward

    # d(sum(a*b))/da = b
    assert_in_delta 4.0, a.grad[0, 0], EPSILON
    assert_in_delta 5.0, a.grad[0, 1], EPSILON
    # d(sum(a*b))/db = a
    assert_in_delta 2.0, b.grad[0, 0], EPSILON
    assert_in_delta 3.0, b.grad[0, 1], EPSILON
  end

  def test_gradient_of_dot_matches_numerical
    a = Tensor.from([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor.from([[0.5, 1.5], [2.5, 3.5]])

    expr = -> { Tensor.from(a.data.dup).dot(Tensor.from(b.data.dup)).sum }
    out  = a.dot(b).sum
    out.backward

    ng_a00 = numerical_grad(a, [0, 0], expr)
    ng_a10 = numerical_grad(a, [1, 0], expr)
    ng_b01 = numerical_grad(b, [0, 1], expr)

    assert_close ng_a00, a.grad[0, 0], "grad a[0,0]"
    assert_close ng_a10, a.grad[1, 0], "grad a[1,0]"
    assert_close ng_b01, b.grad[0, 1], "grad b[0,1]"
  end

  def test_gradient_of_relu
    a = Tensor.from([[2.0, -1.0]])
    a.relu.sum.backward
    assert_in_delta 1.0, a.grad[0, 0], EPSILON  # positive: passes through
    assert_in_delta 0.0, a.grad[0, 1], EPSILON  # negative: blocked
  end

  def test_gradient_of_tanh_matches_numerical
    a = Tensor.from([[0.5, -0.3]])
    expr = -> { Tensor.from(a.data.dup).tanh.sum }
    a.tanh.sum.backward

    ng = numerical_grad(a, [0, 0], expr)
    assert_close ng, a.grad[0, 0], "grad tanh[0,0]"
  end

  def test_broadcast_gradient_addition
    # a is [1,3], b is [2,3] — b broadcasts along axis 0
    a   = Tensor.from([[1.0, 2.0, 3.0]])
    b   = Tensor.from([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    out = (a + b).sum
    out.backward

    # a's gradient must be summed over the broadcast axis (rows)
    assert_equal [1, 3], a.grad.shape
    assert_in_delta 2.0, a.grad[0, 0], EPSILON  # contributed to 2 rows
    assert_in_delta 2.0, a.grad[0, 1], EPSILON
  end

  def test_gradient_through_reshape
    a   = Tensor.from([[1.0, 2.0, 3.0, 4.0]])
    out = a.reshape(2, 2).sum
    out.backward
    assert_equal [1, 4], a.grad.shape
    assert_in_delta 1.0, a.grad[0, 0], EPSILON
  end

  def test_gradient_through_transpose
    a   = Tensor.from([[1.0, 2.0], [3.0, 4.0]])
    out = a.T.sum
    out.backward
    assert_in_delta 1.0, a.grad[0, 0], EPSILON
    assert_in_delta 1.0, a.grad[1, 0], EPSILON
  end

  def test_chained_operations
    # loss = mean((a dot b).relu)
    a = Tensor.from([[1.0, -1.0], [2.0, 0.5]])
    b = Tensor.from([[0.5, 1.0], [-1.0, 2.0]])

    expr = -> {
      Tensor.from(a.data.dup).dot(Tensor.from(b.data.dup)).relu.mean
    }

    out = a.dot(b).relu.mean
    out.backward

    ng_a00 = numerical_grad(a, [0, 0], expr)
    ng_a11 = numerical_grad(a, [1, 1], expr)

    assert_close ng_a00, a.grad[0, 0], "chained grad a[0,0]"
    assert_close ng_a11, a.grad[1, 1], "chained grad a[1,1]"
  end
end
