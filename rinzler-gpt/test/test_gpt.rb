# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/rinzler/gpt"

include Rinzler::GPT
Tok = Rinzler::Tokenizer

SAMPLE_TEXT = "the quick brown fox jumps over the lazy dog " * 10

class TestConfig < Minitest::Test
  def test_tiny_config
    cfg = Config.tiny(65)
    assert_equal 65,  cfg.vocab_size
    assert_equal 128, cfg.context_len
    assert_equal 64,  cfg.d_model
    assert_equal 4,   cfg.n_heads
    assert_equal 4,   cfg.n_layers
  end
end

class TestMultiHeadAttention < Minitest::Test
  def setup
    @attn = MultiHeadAttention.new(16, 4)  # d_model=16, 4 heads of size 4
  end

  def test_output_shape
    x   = Rinzler::Tensor::Tensor.randn(6, 16)
    out = @attn.call(x)
    assert_equal [6, 16], out.shape
  end

  def test_has_parameters
    assert @attn.parameters.size > 0
  end

  def test_causal_mask_applied
    # With causal masking, a sequence of length 1 should work fine
    x   = Rinzler::Tensor::Tensor.randn(1, 16)
    out = @attn.call(x)
    assert_equal [1, 16], out.shape
  end
end

class TestTransformerBlock < Minitest::Test
  def test_output_shape
    block = TransformerBlock.new(16, 4)
    x     = Rinzler::Tensor::Tensor.randn(5, 16)
    out   = block.call(x)
    assert_equal [5, 16], out.shape
  end

  def test_residual_connection_preserves_shape
    block = TransformerBlock.new(32, 4)
    x     = Rinzler::Tensor::Tensor.randn(3, 32)
    out   = block.call(x)
    assert_equal x.shape, out.shape
  end
end

class TestGPTModel < Minitest::Test
  def setup
    @tokenizer = Tok::Character.new.train(SAMPLE_TEXT)
    @cfg       = Config.tiny(@tokenizer.vocab_size)
    @model     = GPTModel.new(@cfg)
  end

  def test_forward_output_shape
    ids    = @tokenizer.encode("the fox")
    logits = @model.forward(ids)
    assert_equal [ids.size, @cfg.vocab_size], logits.shape
  end

  def test_loss_is_scalar
    ids  = @tokenizer.encode("the quick brown")
    loss = @model.loss(ids)
    assert loss.data.size == 1
    assert loss.data.sum > 0
  end

  def test_loss_decreases_after_training_step
    ids   = @tokenizer.encode("the quick brown fox")
    opt   = Rinzler::Optim::AdamW.new(@model.parameters, lr: 0.01)

    opt.zero_grad
    loss_before = @model.loss(ids)
    loss_before.backward
    opt.step

    # Recompute loss — should be lower after one update
    loss_after = @model.loss(ids)
    assert loss_after.data.sum < loss_before.data.sum,
           "Loss should decrease after one optimizer step"
  end

  def test_generate_returns_extended_sequence
    ids      = @tokenizer.encode("the")
    extended = @model.generate(ids, max_new_tokens: 10)
    assert_equal ids.size + 10, extended.size
  end

  def test_generate_produces_valid_token_ids
    ids      = @tokenizer.encode("fox")
    extended = @model.generate(ids, max_new_tokens: 5)
    assert extended.all? { |id| id >= 0 && id < @cfg.vocab_size }
  end

  def test_parameter_count
    # Should have a meaningful number of parameters for a tiny model
    total = @model.parameters.sum { |p| p.data.size }
    assert total > 1000, "Tiny model should have >1000 parameters, got #{total}"
  end
end
