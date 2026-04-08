# frozen_string_literal: true

module Rinzler
  module Optim
    # Scheduler wraps an optimizer and adjusts its learning rate each step
    # according to a schedule. It exposes the same interface as an optimizer
    # so the training loop does not need to change.
    #
    # Usage:
    #   opt       = Opt::AdamW.new(params, lr: 1e-3)
    #   scheduler = Opt::LinearWarmup.new(opt, warmup_steps: 100)
    #
    #   steps.times do
    #     scheduler.zero_grad
    #     loss.backward
    #     scheduler.step   # sets lr for this step, then calls optimizer.step
    #   end
    class Scheduler
      def initialize(optimizer)
        @optimizer  = optimizer
        @step_count = 0
      end

      # Sets the lr for the current step, then delegates to the optimizer.
      def step
        @step_count += 1
        @optimizer.lr = compute_lr(@step_count)
        @optimizer.step
      end

      def zero_grad
        @optimizer.zero_grad
      end

      def lr
        @optimizer.lr
      end

      private

      def compute_lr(_step)
        raise NotImplementedError, "#{self.class} must implement #compute_lr"
      end
    end
  end
end
