# frozen_string_literal: true

module Rinzler
  module GPT
    # MultiHeadAttention — the mechanism that lets every token look at every other token.
    #
    # This is the core insight of "Attention Is All You Need" (Vaswani et al., 2017).
    # Before transformers, sequence models (RNNs, LSTMs) processed tokens one at a time,
    # passing a fixed-size hidden state forward. Information about distant tokens had to
    # survive through many steps — it often didn't.
    #
    # Attention says: let every token directly query every other token.
    # Don't relay information through a bottleneck. Let the model learn which
    # relationships matter, and read them directly.
    #
    # How it works:
    #   Each token produces three vectors from its embedding:
    #     Q (query)  — "what am I looking for?"
    #     K (key)    — "what do I contain?"
    #     V (value)  — "what do I send to whoever attends to me?"
    #
    #   Attention score between token i and token j: dot(Q_i, K_j) / sqrt(d_head)
    #   The sqrt(d_head) scaling prevents dot products from growing large in high dimensions,
    #   which would push softmax into regions of near-zero gradients.
    #
    #   Scores → softmax → attention weights (a probability distribution per token)
    #   Output for token i: weighted sum of all V vectors
    #
    # Multi-head: run this h times in parallel with d_head = d_model/h.
    # Each head can learn to attend to different kinds of relationships
    # (syntax, coreference, proximity, etc.). Their outputs are concatenated
    # and projected back to d_model.
    #
    # Causal mask: for language modeling, token i must not see tokens i+1, i+2...
    # We enforce this by setting future attention scores to -∞ before softmax,
    # which makes those weights exactly 0.
    #
    # Batching: forward accepts either [T, d_model] (single sequence) or
    # [B, T, d_model] (batch). Scores and value aggregation use bmm for the
    # batched case; dot for the single-sequence case.
    class MultiHeadAttention < NN::Module
      def initialize(d_model, n_heads, context_len:)
        raise ArgumentError, "d_model must be divisible by n_heads" unless (d_model % n_heads).zero?

        @d_model = d_model
        @n_heads = n_heads
        @d_head  = d_model / n_heads
        @scale   = 1.0 / Math.sqrt(@d_head)

        # Single projection for Q, K, V together — more efficient than three separate layers.
        # Output is 3*d_model wide; we slice it into thirds.
        @c_attn = NN::Linear.new(d_model, 3 * d_model, bias: true)

        # Output projection — mixes the concatenated head outputs back to d_model
        @c_proj = NN::Linear.new(d_model, d_model, bias: true)

        # Causal mask precomputed once at construction — constant for the life of the model.
        # Previously rebuilt with an O(T²) Ruby loop on every forward pass.
        @causal_mask = build_causal_mask(context_len)
      end

      # x: Tensor [T, d_model] or [B, T, d_model]
      # Returns: Tensor of same shape as x
      def forward(x)
        batched = x.ndim == 3
        seq_len = batched ? x.shape[1] : x.shape[0]

        # Project input to Q, K, V all at once: [..., 3*d_model]
        # Linear handles the 3D case by flattening leading dims internally.
        qkv = @c_attn.call(x)

        # Slice into Q, K, V — each [..., d_model]
        q = qkv.slice_cols(0,            @d_model)
        k = qkv.slice_cols(@d_model,     @d_model)
        v = qkv.slice_cols(@d_model * 2, @d_model)

        # Run each head independently, collect outputs
        head_outputs = (0...@n_heads).map do |h|
          offset = h * @d_head

          # Each head gets a [..., d_head] slice
          q_h = q.slice_cols(offset, @d_head)
          k_h = k.slice_cols(offset, @d_head)
          v_h = v.slice_cols(offset, @d_head)

          if batched
            # scores: [B, T, T] via batched matmul
            # q_h: [B, T, d_head], k_h.transpose_last2: [B, d_head, T]
            scores = q_h.bmm(k_h.transpose_last2) * @scale
          else
            # scores: [T, T]
            scores = q_h.dot(k_h.T) * @scale
          end

          # Causal mask: future tokens become -1e9 so softmax zeroes them out.
          # The [T,T] mask broadcasts to [B,T,T] via numo's trailing-dim broadcast.
          scores = apply_causal_mask(scores, seq_len)

          # Softmax operates on the last axis — attention weights per query token.
          weights = scores.softmax

          if batched
            weights.bmm(v_h)   # [B, T, T] x [B, T, d_head] → [B, T, d_head]
          else
            weights.dot(v_h)   # [T, T] x [T, d_head] → [T, d_head]
          end
        end

        # Concatenate all heads along last axis: [..., d_model]
        concat = Rinzler::Tensor::Tensor.concat_cols(head_outputs)

        # Mix head outputs together
        @c_proj.call(concat)
      end

      private

      # Add -1e9 to all positions (i, j) where j > i (future tokens).
      # Slices the precomputed mask for shorter sequences (e.g. during generation).
      # Broadcasts to [B, T, T] when added to batched scores.
      def apply_causal_mask(scores, seq_len)
        mask = if seq_len == @causal_mask.shape[0]
          @causal_mask
        else
          Rinzler::Tensor::Tensor.new(@causal_mask.data[0...seq_len, 0...seq_len])
        end
        scores + mask
      end

      def build_causal_mask(max_len)
        data = Numo::DFloat.zeros(max_len, max_len)
        max_len.times do |i|
          (i + 1...max_len).each { |j| data[i, j] = -1e9 }
        end
        Rinzler::Tensor::Tensor.new(data)
      end
    end
  end
end
