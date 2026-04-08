# frozen_string_literal: true

module Rinzler
  module Optim
    # SGD with Momentum (Polyak, 1964)
    #
    # Momentum solves SGD's ravine problem by giving the optimizer a "velocity" —
    # it accumulates a running average of past gradients and moves in that
    # averaged direction instead of the raw gradient.
    #
    # Update rule:
    #   v = β * v + (1 - β) * ∇L    (accumulate velocity)
    #   θ = θ - lr * v               (step in velocity direction)
    #
    # (Some formulations use v = β*v + ∇L without the (1-β) scaling —
    # both work, they just absorb the scale difference into lr.)
    #
    # The intuition: imagine rolling a ball down a hilly landscape.
    # SGD is a ball with no mass — it changes direction instantly at every bump.
    # Momentum gives the ball mass. It builds up speed in consistent directions
    # and resists sharp reversals, smoothing out the oscillations that plague SGD.
    #
    # β = 0.9 is the standard default: each step is 90% previous velocity,
    # 10% new gradient information.
    #
    # Why it still falls short:
    #   The learning rate is still global. Sparse gradients (rare features,
    #   embeddings) receive tiny updates while dense ones race ahead.
    #   RMSprop addresses this.
    class SGDMomentum < Optimizer
      def initialize(parameters, lr:, momentum: 0.9, weight_decay: 0.0)
        super(parameters, lr:)
        @momentum     = momentum
        @weight_decay = weight_decay
        # Velocity buffers — one per parameter, initialized to zero
        @velocity     = @parameters.map { |p| Numo::DFloat.zeros(*p.shape) }
      end

      def step
        @parameters.each_with_index do |p, i|
          grad = p.grad
          grad = grad + @weight_decay * p.data if @weight_decay > 0

          @velocity[i] = @momentum * @velocity[i] + (1.0 - @momentum) * grad
          p.data       = p.data - @lr * @velocity[i]
        end
      end
    end
  end
end
