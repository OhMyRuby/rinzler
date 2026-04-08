# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/rinzler/nn"

include Rinzler::NN
T = Rinzler::Tensor::Tensor

class TestParameter < Minitest::Test
  def test_is_a_tensor
    p = Parameter.new(Numo::DFloat[[1.0, 2.0]])
    assert_kind_of Rinzler::Tensor::Tensor, p
  end

  def test_has_grad
    p = Parameter.new(Numo::DFloat[[1.0, 2.0]])
    assert_equal [1, 2], p.grad.shape
  end
end

class TestModule < Minitest::Test
  def test_collects_parameters
    layer = Linear.new(3, 2)
    params = layer.parameters
    assert_equal 2, params.size   # weight + bias
    assert params.all? { |p| p.is_a?(Parameter) }
  end

  def test_no_bias_collects_one_parameter
    layer = Linear.new(3, 2, bias: false)
    assert_equal 1, layer.parameters.size
  end

  def test_zero_grad_resets_gradients
    layer = Linear.new(2, 2)
    layer.parameters.each { |p| p.grad = Numo::DFloat.ones(*p.shape) }
    layer.zero_grad
    layer.parameters.each do |p|
      assert_equal 0.0, p.grad.sum
    end
  end
end

class TestLinear < Minitest::Test
  EPSILON = 1e-5
  DELTA   = 1e-3

  def test_output_shape
    layer = Linear.new(4, 8)
    x   = T.randn(2, 4)
    out = layer.call(x)
    assert_equal [2, 8], out.shape
  end

  def test_no_bias_output_shape
    layer = Linear.new(4, 8, bias: false)
    x   = T.randn(2, 4)
    out = layer.call(x)
    assert_equal [2, 8], out.shape
  end

  def test_gradient_flows_to_weight
    layer = Linear.new(3, 2, bias: false)
    x     = T.from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out   = layer.call(x).sum
    out.backward
    # Weight grad should be non-zero
    assert layer.weight.grad.sum.abs > 0
  end

  def test_gradient_flows_to_input
    layer = Linear.new(3, 2)
    x     = T.from([[1.0, 2.0, 3.0]])
    out   = layer.call(x).sum
    out.backward
    assert_equal [1, 3], x.grad.shape
  end

  def test_gradient_flows_to_bias
    layer = Linear.new(3, 2)
    x     = T.from([[1.0, 2.0, 3.0]])
    out   = layer.call(x).sum
    out.backward
    # Each bias element received gradient = 1 (sum loss)
    assert_in_delta 1.0, layer.bias.grad[0], EPSILON
    assert_in_delta 1.0, layer.bias.grad[1], EPSILON
  end
end

class TestEmbedding < Minitest::Test
  EPSILON = 1e-5

  def test_output_shape
    emb = Embedding.new(10, 4)
    out = emb.call([0, 3, 7])
    assert_equal [3, 4], out.shape
  end

  def test_selects_correct_rows
    emb = Embedding.new(5, 3)
    # Manually set known weights
    emb.weight.data[2, true] = Numo::DFloat[1.0, 2.0, 3.0]
    out = emb.call([2])
    assert_in_delta 1.0, out.data[0, 0], EPSILON
    assert_in_delta 2.0, out.data[0, 1], EPSILON
    assert_in_delta 3.0, out.data[0, 2], EPSILON
  end

  def test_gradient_scatters_to_selected_rows
    emb = Embedding.new(5, 3)
    out = emb.call([1, 3])
    out.backward
    # Rows 1 and 3 should have nonzero grad; others should be zero
    assert emb.weight.grad[1, true].sum > 0
    assert emb.weight.grad[3, true].sum > 0
    assert_in_delta 0.0, emb.weight.grad[0, true].sum, EPSILON
    assert_in_delta 0.0, emb.weight.grad[2, true].sum, EPSILON
  end

  def test_repeated_index_accumulates_gradient
    emb = Embedding.new(5, 2)
    # Token 0 appears twice — its gradient should double
    out = emb.call([0, 0])
    out.backward
    single = Numo::DFloat.ones(1, 2)
    # 2 appearances × embedding_dim(2) × grad(1.0) = 4.0
    assert_in_delta 4.0, emb.weight.grad[0, true].sum, 1e-4
  end
end

class TestLayerNorm < Minitest::Test
  EPSILON = 1e-5
  DELTA   = 1e-3

  def test_output_shape
    ln  = LayerNorm.new(4)
    x   = T.randn(3, 4)
    out = ln.call(x)
    assert_equal [3, 4], out.shape
  end

  def test_output_is_normalized
    ln  = LayerNorm.new(8)
    x   = T.from([Array.new(8) { |i| i.to_f * 10 }])
    out = ln.call(x)
    mean = out.data[0, true].sum / 8
    var  = ((out.data[0, true] - mean) ** 2).sum / 8
    assert_in_delta 0.0, mean, DELTA
    assert_in_delta 1.0, var,  DELTA
  end

  def test_gradient_flows_to_input
    ln     = LayerNorm.new(4)
    x      = T.randn(2, 4)
    # Weighted sum breaks the symmetry that makes sum(layernorm(x)) invariant.
    # With uniform upstream gradient, LayerNorm's dx is exactly 0 by design.
    weights = T.from([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    out     = (ln.call(x) * weights).sum
    out.backward
    assert_equal [2, 4], x.grad.shape
    assert x.grad.abs.sum > 0
  end

  def test_gradient_flows_to_weight_and_bias
    ln  = LayerNorm.new(4)
    x   = T.randn(3, 4)
    out = ln.call(x).sum
    out.backward
    assert ln.weight.grad.sum.abs > 0
    # bias.grad = dy.sum(axis:0), shape [4]. Each of 4 features gets batch(3) × 1.0 = 3.0
    assert_in_delta 3.0, ln.bias.grad[0], DELTA
    assert_in_delta 3.0, ln.bias.grad[1], DELTA
  end

  def test_numerical_gradient_of_layer_norm
    ln = LayerNorm.new(3)
    x  = T.from([[0.5, -0.3, 1.2], [0.1, 0.9, -0.4]])

    ln.call(x).sum.backward
    analytical = x.grad[0, 0]

    h = 1e-4
    x.data[0, 0] += h
    fph = ln.call(T.new(x.data.dup)).sum.data.sum
    x.data[0, 0] -= 2 * h
    fmh = ln.call(T.new(x.data.dup)).sum.data.sum
    x.data[0, 0] += h
    numerical = (fph - fmh) / (2 * h)

    assert_in_delta numerical, analytical, DELTA
  end
end
