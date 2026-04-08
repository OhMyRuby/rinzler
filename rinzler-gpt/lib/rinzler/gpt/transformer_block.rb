# frozen_string_literal: true

module Rinzler
  module GPT
    # TransformerBlock — one layer of the GPT stack.
    #
    # A transformer is just N of these stacked on top of each other.
    # GPT-1 had 12. GPT-2 small had 12, GPT-2 XL had 48. GPT-3 had 96.
    # Each block adds depth — more layers = more capacity to learn abstractions.
    #
    # Structure (Pre-LN variant, used by GPT-2 and most modern transformers):
    #
    #   x → LayerNorm → MultiHeadAttention → + x  (residual)
    #     → LayerNorm → FeedForward         → + x  (residual)
    #
    # Two key design decisions worth understanding:
    #
    # 1. Residual connections (He et al., 2016 — ResNets)
    #    output = x + sublayer(x), not just output = sublayer(x)
    #
    #    This is why deep networks became trainable. Without residuals, gradients
    #    vanish or explode through many layers. With residuals, there's always a
    #    direct gradient highway back through the addition operation. The network
    #    learns *corrections* to the identity rather than full transformations.
    #    At initialization, the residual path dominates; as training proceeds,
    #    the sublayers contribute more.
    #
    # 2. Feed-Forward Network (FFN)
    #    Two linear layers with a GELU activation:
    #      FFN(x) = W₂ · GELU(W₁ · x + b₁) + b₂
    #
    #    The inner dimension is typically 4x d_model (e.g., 768 → 3072 for GPT-2).
    #    This is where most of the model's "memory" lives. Attention routes
    #    information; FFN processes and stores it. Recent mechanistic interpretability
    #    research (Geva et al., 2021) suggests FFN layers function as key-value
    #    memories — each neuron fires for specific patterns and emits associated facts.
    #
    # 3. Pre-LN vs Post-LN
    #    Original transformer: normalize *after* the sublayer (Post-LN).
    #    GPT-2 onward: normalize *before* (Pre-LN). Pre-LN trains more stably
    #    because the residual stream stays at a consistent scale throughout.
    GELU_COEF = Math.sqrt(2.0 / Math::PI)  # ≈ 0.7978845608

    class TransformerBlock < NN::Module
      def initialize(d_model, n_heads, ffn_mult: 4, context_len:)
        @ln1      = NN::LayerNorm.new(d_model)
        @attn     = MultiHeadAttention.new(d_model, n_heads, context_len:)
        @ln2      = NN::LayerNorm.new(d_model)

        # FFN: expand → activate → contract
        d_ffn = d_model * ffn_mult
        @fc1  = NN::Linear.new(d_model, d_ffn)
        @fc2  = NN::Linear.new(d_ffn, d_model)
      end

      # x: Tensor [seq_len, d_model]
      def forward(x)
        # Attention sub-layer with residual
        x = x + @attn.call(@ln1.call(x))

        # FFN sub-layer with residual
        # GELU (Hendrycks & Gimpel, 2016) — the correct activation for GPT.
        # Tanh approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        # This is what GPT-2 uses. Smoother than ReLU in the negative region,
        # which matters for gradient flow through deep FFN stacks.
        h     = @fc1.call(@ln2.call(x))
        inner = (h + h ** 3 * 0.044715) * GELU_COEF
        h     = h * inner.tanh * 0.5 + h * 0.5
        x     = x + @fc2.call(h)

        x
      end
    end
  end
end
