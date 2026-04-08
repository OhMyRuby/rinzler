# frozen_string_literal: true

module Rinzler
  module Optim
    # CosineWithWarmup — linear warmup then cosine decay to min_lr.
    #
    # Schedule:
    #   step <= warmup_steps : lr = base_lr * (step / warmup_steps)
    #   step >  warmup_steps : lr = min_lr + 0.5 * (base_lr - min_lr) *
    #                               (1 + cos(π * progress))
    #
    # where progress = (step - warmup_steps) / (total_steps - warmup_steps),
    # clamped to [0, 1].
    #
    # Cosine decay is smoother than linear — it starts fast (steep early
    # gradient), slows as it approaches min_lr, and never goes below it.
    # This matches how most modern language model training schedules work
    # (GPT-3, LLaMA, etc. all use variants of this).
    class CosineWithWarmup < Scheduler
      def initialize(optimizer, warmup_steps:, total_steps:, min_lr: 0.0)
        super(optimizer)
        @base_lr      = optimizer.lr
        @warmup_steps = warmup_steps
        @total_steps  = total_steps
        @min_lr       = min_lr
      end

      private

      def compute_lr(step)
        if step <= @warmup_steps
          return @base_lr * step.to_f / @warmup_steps
        end

        progress = [(step - @warmup_steps).to_f / (@total_steps - @warmup_steps), 1.0].min
        @min_lr + 0.5 * (@base_lr - @min_lr) * (1.0 + Math.cos(Math::PI * progress))
      end
    end
  end
end
