# frozen_string_literal: true

module Rinzler
  module Optim
    # Adam — Adaptive Moment Estimation (Kingma & Ba, 2014)
    #
    # Adam combines the two key ideas that preceded it:
    #   - Momentum  (from SGD+Momentum): smooth gradient direction with a running average
    #   - Adaptive rates (from RMSprop): scale steps by inverse RMS of past gradients
    #
    # It tracks two moments per parameter:
    #   m = β₁ * m + (1 - β₁) * ∇L        (1st moment: mean of gradients — direction)
    #   v = β₂ * v + (1 - β₂) * ∇L²       (2nd moment: mean of squared gradients — scale)
    #
    # Bias correction: at the start of training, m and v are initialized to zero.
    # They're biased toward zero for the first few steps before enough gradients
    # have accumulated. The correction inflates the estimates early on:
    #
    #   m̂ = m / (1 - β₁ᵗ)
    #   v̂ = v / (1 - β₂ᵗ)
    #
    # Final update:
    #   θ = θ - lr * m̂ / (√v̂ + ε)
    #
    # Defaults (β₁=0.9, β₂=0.999) come from the original paper and work well
    # across a staggering range of tasks. Adam became the default optimizer for
    # deep learning almost overnight after publication.
    #
    # Why it still falls short — one specific problem:
    #   Weight decay (L2 regularization) interacts incorrectly with adaptive rates.
    #   In SGD, weight decay is equivalent to L2 reg. In Adam, the adaptive scaling
    #   distorts the regularization — parameters with large gradients get less
    #   regularization than parameters with small gradients, which is backwards.
    #   AdamW fixes this.
    class Adam < Optimizer
      def initialize(parameters, lr:, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0)
        super(parameters, lr:)
        @beta1        = beta1
        @beta2        = beta2
        @eps          = eps
        @weight_decay = weight_decay
        @step_count   = 0

        @m = @parameters.map { |p| Numo::DFloat.zeros(*p.shape) }  # 1st moment
        @v = @parameters.map { |p| Numo::DFloat.zeros(*p.shape) }  # 2nd moment
      end

      def step
        @step_count += 1
        t = @step_count

        # Bias correction factors grow toward 1.0 as t increases
        bc1 = 1.0 - @beta1 ** t
        bc2 = 1.0 - @beta2 ** t

        @parameters.each_with_index do |p, i|
          grad = p.grad

          # L2 penalty folded into gradient (broken — see AdamW for the fix)
          grad = grad + @weight_decay * p.data if @weight_decay > 0

          @m[i] = @beta1 * @m[i] + (1.0 - @beta1) * grad
          @v[i] = @beta2 * @v[i] + (1.0 - @beta2) * grad ** 2

          m_hat = @m[i] / bc1
          v_hat = @v[i] / bc2

          p.data = p.data - @lr * m_hat / (Numo::NMath.sqrt(v_hat) + @eps)
        end
      end
    end
  end
end
