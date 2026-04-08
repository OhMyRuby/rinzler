# frozen_string_literal: true

module Rinzler
  module Optim
    # RMSprop — Root Mean Square Propagation (Hinton, 2012)
    #
    # Never formally published — introduced in a Coursera lecture slide.
    # One of the most influential unpublished ideas in deep learning.
    #
    # The insight: different parameters have wildly different gradient magnitudes.
    # A weight in a dense layer might see gradients of 0.5 every step.
    # An embedding for a rare word might see gradients of 0.001 once per epoch.
    # Using the same learning rate for both is absurd.
    #
    # RMSprop fixes this with a per-parameter adaptive learning rate.
    # It tracks a running average of squared gradients (v), then divides
    # the update by the root of that average:
    #
    #   v = β * v + (1 - β) * ∇L²       (mean squared gradient, per parameter)
    #   θ = θ - (lr / √(v + ε)) * ∇L    (scale step by inverse RMS)
    #
    # Parameters with large historical gradients get smaller steps.
    # Parameters with small historical gradients get larger steps.
    # The optimizer self-tunes the effective learning rate per parameter.
    #
    # ε (epsilon) is a small constant (~1e-8) to prevent division by zero
    # when a gradient has been near zero for a long time.
    #
    # Why it still falls short:
    #   It adapts the scale but not the direction. Gradients are still noisy,
    #   and there's no momentum to smooth the path. Adam combines both.
    class RMSprop < Optimizer
      def initialize(parameters, lr:, beta: 0.99, eps: 1e-8, weight_decay: 0.0)
        super(parameters, lr:)
        @beta         = beta
        @eps          = eps
        @weight_decay = weight_decay
        # Running mean of squared gradients — one buffer per parameter
        @sq_avg       = @parameters.map { |p| Numo::DFloat.zeros(*p.shape) }
      end

      def step
        @parameters.each_with_index do |p, i|
          grad = p.grad
          grad = grad + @weight_decay * p.data if @weight_decay > 0

          @sq_avg[i] = @beta * @sq_avg[i] + (1.0 - @beta) * grad ** 2
          p.data     = p.data - (@lr / (Numo::NMath.sqrt(@sq_avg[i]) + @eps)) * grad
        end
      end
    end
  end
end
