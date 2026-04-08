# frozen_string_literal: true

module Rinzler
  module NN
    # Linear implements a fully connected layer: y = x · Wᵀ + b
    #
    # Weight shape: [out_features, in_features]
    # Bias shape:   [out_features]
    # Input shape:  [batch, in_features]  →  Output: [batch, out_features]
    #
    # Why Wᵀ? We store W as [out, in] (rows = output neurons, cols = inputs)
    # so each row is one neuron's weight vector. To apply it to a batch input
    # [batch, in], we compute x · Wᵀ which gives [batch, out]. This is the
    # standard convention (same as PyTorch's nn.Linear).
    #
    # Weight initialization uses Kaiming uniform (He init), which keeps
    # activation variance stable through deep ReLU networks.
    class Linear < Module
      attr_reader :weight, :bias

      def initialize(in_features, out_features, bias: true)
        @in_features  = in_features
        @out_features = out_features

        # Kaiming uniform: std = sqrt(2 / in_features)
        # Scaled to uniform [-bound, bound] where bound = sqrt(3) * std
        bound   = Math.sqrt(1.0 / in_features)
        w_data  = Numo::DFloat.new(out_features, in_features).rand * (2 * bound) - bound
        @weight = Parameter.new(w_data)

        if bias
          b_data = Numo::DFloat.new(out_features).rand * (2 * bound) - bound
          @bias  = Parameter.new(b_data)
        end
      end

      def forward(x)
        # x: [batch, in_features] or [batch, seq, in_features]
        # weight: [out_features, in_features]
        # x.dot(weight.T) → [..., out_features]
        if x.ndim > 2
          # Flatten leading dims, apply linear, reshape back.
          # This lets the same Linear layer handle both 2D and 3D input
          # without any architectural changes — used by transformers where
          # x is [batch, seq, features].
          leading = x.shape[0...-1]
          flat    = x.reshape(leading.reduce(:*), @in_features)
          result  = flat.dot(@weight.T)
          result  = result + @bias if @bias
          result.reshape(*leading, @out_features)
        else
          out = x.dot(@weight.T)
          @bias ? out + @bias : out
        end
      end
    end
  end
end
