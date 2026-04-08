# frozen_string_literal: true

module Rinzler
  module Optim
    # AdamW — Adam with Decoupled Weight Decay (Loshchilov & Hutter, 2017)
    #
    # The fix for Adam's weight decay bug. Simple in retrospect, took years to notice.
    #
    # The problem with Adam + L2 regularization:
    #   Adam scales every gradient update by 1/√v̂. When you add weight decay
    #   as an L2 penalty (grad += λ*θ), that penalty also gets scaled down
    #   for parameters with large gradient history. The result: parameters that
    #   are updated frequently (common features) get less regularization than
    #   parameters updated rarely. That's the opposite of what you want.
    #
    # The fix — decouple weight decay from the gradient entirely:
    #   Instead of folding λ*θ into the gradient before the Adam update,
    #   apply it as a direct parameter shrinkage *after* the Adam step:
    #
    #   θ = θ - lr * m̂/(√v̂+ε)    [Adam update, no weight decay in gradient]
    #   θ = θ - lr * λ * θ         [weight decay applied directly, no adaptive scaling]
    #
    #   Which simplifies to:
    #   θ = θ * (1 - lr * λ) - lr * m̂/(√v̂+ε)
    #
    # This ensures every parameter gets the same proportional shrinkage per step,
    # regardless of its gradient history. That's what L2 regularization is
    # supposed to do.
    #
    # AdamW is the optimizer behind GPT-2, GPT-3, and most modern transformers.
    # If you're training a language model, this is your starting point.
    class AdamW < Optimizer
      def initialize(parameters, lr:, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01)
        super(parameters, lr:)
        @beta1        = beta1
        @beta2        = beta2
        @eps          = eps
        @weight_decay = weight_decay
        @step_count   = 0

        @m = @parameters.map { |p| Numo::DFloat.zeros(*p.shape) }
        @v = @parameters.map { |p| Numo::DFloat.zeros(*p.shape) }
      end

      # Serialize moment buffers and step count for checkpoint saving.
      # Returns Numo arrays directly — callers write them to binary.
      def checkpoint_state
        {
          "step_count" => @step_count,
          "m"          => @m,
          "v"          => @v
        }
      end

      # Restore from a checkpoint_state hash.
      # Accepts Numo arrays (new binary format) or Ruby arrays (legacy JSON format).
      def load_checkpoint_state!(state)
        @step_count = state["step_count"]
        @m = state["m"].zip(@parameters).map { |data, p| Numo::DFloat.cast(data).reshape(*p.shape) }
        @v = state["v"].zip(@parameters).map { |data, p| Numo::DFloat.cast(data).reshape(*p.shape) }
      end

      def step
        @step_count += 1
        t = @step_count

        bc1 = 1.0 - @beta1 ** t
        bc2 = 1.0 - @beta2 ** t

        @parameters.each_with_index do |p, i|
          grad = p.grad  # raw gradient — weight decay NOT added here

          @m[i] = @beta1 * @m[i] + (1.0 - @beta1) * grad
          @v[i] = @beta2 * @v[i] + (1.0 - @beta2) * grad ** 2

          m_hat = @m[i] / bc1
          v_hat = @v[i] / bc2

          # Decoupled weight decay: direct shrinkage, bypasses adaptive scaling
          p.data = p.data * (1.0 - @lr * @weight_decay) \
                 - @lr * m_hat / (Numo::NMath.sqrt(v_hat) + @eps)
        end
      end
    end
  end
end
