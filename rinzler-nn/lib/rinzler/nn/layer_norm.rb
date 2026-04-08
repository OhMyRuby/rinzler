# frozen_string_literal: true

module Rinzler
  module NN
    # LayerNorm normalizes each sample independently across its features.
    #
    # For an input x of shape [batch, features]:
    #   1. Compute mean and variance across the feature dimension (per row)
    #   2. Normalize: x_norm = (x - mean) / sqrt(var + eps)
    #   3. Scale and shift with learned parameters: y = weight * x_norm + bias
    #
    # Why LayerNorm instead of BatchNorm for transformers?
    # BatchNorm normalizes across the batch dimension — it needs a large batch
    # to get stable statistics and behaves differently at train vs inference.
    # LayerNorm normalizes per-sample, making it batch-independent and much
    # more stable for sequence models where batch sizes vary.
    #
    # We implement this with a custom backward rather than composing primitives.
    # The analytic gradient is numerically stable and avoids building a long
    # chain through mean/var/sqrt operations in the autograd graph.
    class LayerNorm < Module
      attr_reader :weight, :bias

      def initialize(normalized_shape)
        @normalized_shape = Array(normalized_shape)

        # weight (γ) starts at 1 — no scaling initially
        # bias   (β) starts at 0 — no shift initially
        @weight = Parameter.new(Numo::DFloat.ones(*@normalized_shape))
        @bias   = Parameter.new(Numo::DFloat.zeros(*@normalized_shape))
        @eps    = 1e-5
      end

      # x: Tensor [batch, features] or [batch, seq, features]
      # Normalizes over the last dimension for each position independently.
      def forward(x)
        original_shape = x.shape
        features       = original_shape[-1]
        batch          = original_shape[0...-1].reduce(1, :*)

        # Flatten to [batch, features] for uniform row-wise processing.
        # For 2D input this is a no-op; for 3D it merges batch and seq dims.
        x_2d   = x.data.reshape(batch, features)
        mean   = row_mean(x_2d, batch, features)
        diff   = x_2d - mean
        var    = row_mean(diff ** 2, batch, features)
        std    = Numo::NMath.sqrt(var + @eps)
        x_norm = diff / std                            # [batch, features]

        # y = γ * x_norm + β, then restore original shape
        y_data = (@weight.data * x_norm + @bias.data).reshape(*original_shape)
        out    = Rinzler::Tensor::Tensor.new(y_data, children: [x, @weight, @bias], op: :layer_norm)

        # Backward through LayerNorm (analytic form):
        #
        # Let N = features, x̂ = x_norm
        # dL/dx = (γ/std) * (dL/dy - mean(dL/dy) - x̂ * mean(dL/dy * x̂))
        #
        # dL/dγ = sum(dL/dy * x̂, axis=0)  [across all batch positions]
        # dL/dβ = sum(dL/dy,     axis=0)
        out._set_backward do
          # out.grad has original_shape — flatten for row-wise operations
          dy    = out.grad.reshape(batch, features)    # [batch, features]
          dy_xn = dy * x_norm                          # [batch, features]

          mean_dy    = row_mean(dy,    batch, features)
          mean_dy_xn = row_mean(dy_xn, batch, features)

          dx = (@weight.data / std) * (dy - mean_dy - x_norm * mean_dy_xn)

          x.grad       = x.grad       + dx.reshape(*original_shape)
          @weight.grad = @weight.grad + dy_xn.sum(0)
          @bias.grad   = @bias.grad   + dy.sum(0)
        end

        out
      end

      private

      # Sum each row then divide by N, result shape [batch, 1] for broadcasting
      def row_mean(arr, batch, features)
        arr.sum(1).reshape(batch, 1) / features.to_f
      end
    end
  end
end
