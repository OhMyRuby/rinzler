# frozen_string_literal: true

module Rinzler
  module NN
    # Module is the base class for all neural network layers.
    #
    # It provides three things:
    #   1. Parameter discovery — walk instance variables to find all Parameters,
    #      including those nested inside child Modules or Arrays of Modules.
    #   2. zero_grad — reset all parameter gradients to zero before each step.
    #   3. call → forward — the standard interface for running a layer.
    #
    # To define a layer, subclass Module and implement `forward`.
    class Module
      # Collect every Parameter this module (and any child modules) owns.
      # This is how the optimizer finds what to update without requiring
      # explicit registration — just assign Parameters as instance variables.
      def parameters
        instance_variables.flat_map do |var|
          val = instance_variable_get(var)
          case val
          when Parameter     then [val]
          when Module        then val.parameters
          when Array         then val.flat_map { collect_params(it) }
          else []
          end
        end
      end

      # Zero out all gradients. Call this before each forward/backward pass
      # so gradients don't accumulate across batches.
      def zero_grad
        parameters.each { |p| p.grad = Numo::DFloat.zeros(*p.shape) }
      end

      # The standard call interface. Subclasses implement `forward`.
      def call(*args) = forward(*args)

      def forward(*)
        raise NotImplementedError, "#{self.class} must implement #forward"
      end

      def inspect = "#<#{self.class.name} parameters=#{parameters.size}>"

      private

      def collect_params(val)
        case val
        when Parameter then [val]
        when Module    then val.parameters
        when Array     then val.flat_map { collect_params(it) }
        else []
        end
      end
    end
  end
end
