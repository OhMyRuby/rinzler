# frozen_string_literal: true

module Rinzler
  module Optim
    # Optimizer is the base class for all gradient-based parameter update rules.
    #
    # The training loop is always the same three steps:
    #   1. Forward pass  — compute loss
    #   2. Backward pass — compute all gradients (autograd)
    #   3. Step          — update parameters using those gradients (optimizer)
    #
    # The optimizer only owns step 3. Everything before it is the model's job.
    # zero_grad clears the gradients so they don't accumulate across batches.
    class Optimizer
      attr_accessor :lr

      def initialize(parameters, lr:)
        @parameters = Array(parameters)
        @lr         = lr
      end

      def step
        raise NotImplementedError, "#{self.class} must implement #step"
      end

      def zero_grad
        @parameters.each { |p| p.grad.fill(0.0) }
      end

      # Clip gradients by global L2 norm.
      #
      # Computes the norm across all parameters, then rescales every gradient
      # uniformly so the global norm does not exceed max_norm. This prevents
      # occasional large gradient spikes from destabilizing the optimizer's
      # moment estimates without changing the gradient direction.
      #
      # Returns the pre-clip global norm (useful for logging).
      def clip_grad_norm!(max_norm)
        total_sq = @parameters.sum { |p| (p.grad ** 2).sum }
        norm     = Math.sqrt(total_sq)
        coef     = max_norm / [norm, 1e-6].max
        @parameters.each { |p| p.grad = p.grad * coef } if coef < 1.0
        norm
      end
    end
  end
end
