# frozen_string_literal: true

module Rinzler
  module Optim
    # LinearWarmup — ramp lr from 0 to base_lr over warmup_steps, then hold.
    #
    # Schedule:
    #   step <= warmup_steps : lr = base_lr * (step / warmup_steps)
    #   step >  warmup_steps : lr = base_lr
    #
    # Why warmup matters: at the start of training the optimizer's moment
    # estimates (m, v in Adam) are initialized to zero and haven't converged
    # yet. A full lr on step 1 drives large, noisy updates before the
    # optimizer knows which direction to trust. Warmup delays full-speed
    # updates until the estimates have accumulated enough signal.
    class LinearWarmup < Scheduler
      def initialize(optimizer, warmup_steps:)
        super(optimizer)
        @base_lr      = optimizer.lr
        @warmup_steps = warmup_steps
      end

      private

      def compute_lr(step)
        return @base_lr * step.to_f / @warmup_steps if step <= @warmup_steps

        @base_lr
      end
    end
  end
end
