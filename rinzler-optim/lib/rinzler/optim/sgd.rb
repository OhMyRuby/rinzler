# frozen_string_literal: true

module Rinzler
  module Optim
    # SGD — Stochastic Gradient Descent (Robbins & Monro, 1951)
    #
    # The oldest optimizer and the one everything else builds on.
    # The rule is almost insultingly simple:
    #
    #   θ = θ - lr * ∇L
    #
    # Move each parameter a small step in the direction that reduces loss.
    # "Stochastic" means we compute the gradient on a random mini-batch
    # rather than the full dataset — noisy, but fast and often generalizes better.
    #
    # Why it fails:
    #   - One learning rate for every parameter, forever. A parameter whose
    #     gradient is always tiny (rare feature) learns at the same rate as
    #     one whose gradient is huge (common feature). That's wrong.
    #   - Ravines: if the loss surface curves sharply in one direction and
    #     gently in another, SGD oscillates across the sharp dimension while
    #     crawling along the gentle one. Momentum fixes this.
    #   - Sensitive to learning rate choice. Too high: diverges. Too low: takes forever.
    #
    # Despite all that, SGD is still competitive for vision models when tuned well.
    # For language models, Adam-family optimizers dominate.
    class SGD < Optimizer
      def initialize(parameters, lr:, weight_decay: 0.0)
        super(parameters, lr:)
        @weight_decay = weight_decay
      end

      def step
        @parameters.each do |p|
          grad = p.grad
          grad = grad + @weight_decay * p.data if @weight_decay > 0
          p.data = p.data - @lr * grad
        end
      end
    end
  end
end
