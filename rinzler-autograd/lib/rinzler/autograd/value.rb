# frozen_string_literal: true

require "set"

module Rinzler
  module Autograd
    # Value wraps a single scalar number and tracks everything needed to
    # compute gradients through it via reverse-mode backpropagation.
    #
    # Every arithmetic operation on a Value produces a new Value that
    # remembers two things:
    #   1. Which Values produced it (its "children" in the computation graph)
    #   2. How to distribute incoming gradient back to those children (its "_backward" proc)
    #
    # When you call .backward on the final output (e.g. loss), it walks
    # the graph in reverse and accumulates gradients at every node.
    class Value
      attr_accessor :data, :grad
      attr_reader :children, :op

      def initialize(data, children: [], op: nil)
        @data      = data.to_f
        @grad      = 0.0      # starts at zero — no gradient until backward is called
        @children  = children
        @op        = op       # human-readable label for debugging ("*", "+", "tanh", etc.)
        @_backward = -> {}    # no-op by default; set by each operation below
      end

      # ── Arithmetic ────────────────────────────────────────────────────────────

      def +(other)
        other = coerce_value(other)
        out   = Value.new(@data + other.data, children: [self, other], op: :+)

        # The gradient of addition just passes through unchanged to both inputs.
        # If out.grad = 1.0, then self.grad += 1.0 and other.grad += 1.0.
        out._set_backward do
          self.grad  += out.grad
          other.grad += out.grad
        end

        out
      end

      def *(other)
        other = coerce_value(other)
        out   = Value.new(@data * other.data, children: [self, other], op: :*)

        # The gradient of multiplication: d(a*b)/da = b, d(a*b)/db = a.
        # Each input's gradient is scaled by the *other* input's data value.
        out._set_backward do
          self.grad  += other.data * out.grad
          other.grad += @data * out.grad
        end

        out
      end

      def **(exp)
        raise ArgumentError, "exponent must be a numeric constant, not a Value" if exp.is_a?(Value)

        out = Value.new(@data**exp, children: [self], op: :"**#{exp}")

        # Power rule: d(x^n)/dx = n * x^(n-1)
        out._set_backward do
          self.grad += exp * (@data**(exp - 1)) * out.grad
        end

        out
      end

      # Unary negation: -a is just a * -1
      def -@  = self * -1

      # Subtraction and division are defined in terms of the primitives above.
      # This keeps the backward logic in one place per operation.
      def -(other) = self + (-coerce_value(other))
      def /(other) = self * coerce_value(other)**-1

      # ── Activation functions ─────────────────────────────────────────────────
      # These are the non-linearities that let neural networks learn complex patterns.
      # Without them, a stack of linear layers collapses into a single linear layer.

      def tanh
        t   = Math.tanh(@data)
        out = Value.new(t, children: [self], op: :tanh)

        # Derivative of tanh: 1 - tanh(x)^2
        out._set_backward { self.grad += (1.0 - t**2) * out.grad }

        out
      end

      def relu
        out = Value.new([@data, 0].max, children: [self], op: :relu)

        # ReLU gradient: 1 if the input was positive, 0 otherwise.
        # This is the "dead neuron" problem — neurons stuck at 0 get no gradient signal.
        out._set_backward { self.grad += (out.data > 0 ? 1.0 : 0.0) * out.grad }

        out
      end

      def exp
        e   = Math.exp(@data)
        out = Value.new(e, children: [self], op: :exp)

        # e^x is its own derivative.
        out._set_backward { self.grad += e * out.grad }

        out
      end

      def log
        raise ArgumentError, "log of non-positive value: #{@data}" if @data <= 0

        out = Value.new(Math.log(@data), children: [self], op: :log)

        # Derivative of ln(x) = 1/x
        out._set_backward { self.grad += (1.0 / @data) * out.grad }

        out
      end

      # ── Backpropagation ───────────────────────────────────────────────────────

      # Walk the full computation graph in reverse topological order and
      # accumulate gradients at every node.
      #
      # Topological order means: a node only appears after all the nodes
      # that depend on it. Reversing that gives us the backward pass order —
      # gradients flow from output back to inputs.
      def backward
        topo    = []
        visited = Set.new

        build_topo = ->(node) {
          next if visited.include?(node)
          visited.add(node)
          node.children.each { build_topo.call(it) }
          topo << node
        }

        build_topo.call(self)

        # The output node's gradient with respect to itself is always 1.
        # "How much does the loss change if I change the loss? By exactly 1."
        @grad = 1.0

        topo.reverse_each(&:_backward)
      end

      # ── Support ───────────────────────────────────────────────────────────────

      # Allows `2 + value` and `2 * value` — Ruby calls coerce when the left
      # operand doesn't know how to handle a Value.
      def coerce(other) = [Value.new(other), self]

      def to_s    = "Value(data=#{@data.round(4)}, grad=#{@grad.round(4)})"
      def inspect = to_s

      def _set_backward(&) = @_backward = Proc.new(&)
      def _backward        = @_backward.call

      private

      def coerce_value(other) = other.is_a?(Value) ? other : Value.new(other)
    end
  end
end
