# frozen_string_literal: true

require "set"
require "numo/narray"

# Use BLAS-backed dot products for large matrices when numo-linalg is available.
# Below this threshold numo's own C is faster due to BLAS call overhead.
LINALG_THRESHOLD = 256

begin
  require "fiddle"
  require "numo/linalg"
  # Verify BLAS actually loaded — numo-linalg silently patches NArray#dot even
  # if dlopen failed, so we must confirm the library is genuinely available.
  Numo::Linalg.dot(Numo::DFloat[[1.0]], Numo::DFloat[[1.0]].transpose)
  LINALG_AVAILABLE = true
  Rinzler.logger.info("BLAS backend active via numo-linalg — matmul >= #{LINALG_THRESHOLD}d routes to OpenBLAS")
rescue LoadError, RuntimeError
  LINALG_AVAILABLE = false
  Rinzler.logger.info("numo-linalg unavailable — using numo native dot for all matrix ops")
end

module Rinzler
  module Tensor
    # Backend selection: :cpu (default) or :vulkan
    #
    # When set to :vulkan, dot and bmm route their forward pass through the
    # Vulkan GPU compute pipeline. Gradients remain on CPU (autograd is still
    # pure Ruby). This trades slightly higher per-call overhead for dramatically
    # better throughput on large matrix multiplies — the operation that dominates
    # transformer forward passes.
    #
    # Usage:
    #   require "rinzler/vulkan"
    #   Rinzler::Tensor.backend = :vulkan
    class << self
      attr_reader :backend

      def backend=(val)
        raise ArgumentError, "Unknown backend #{val}" unless %i[cpu vulkan].include?(val)

        if val == :vulkan
          require "rinzler/vulkan"
          Rinzler::Vulkan.ensure_initialized!
        end

        @backend = val
      end
    end

    # Default
    @backend = :cpu
    # Tensor wraps a Numo::DFloat n-dimensional array and tracks the
    # computation graph needed for reverse-mode automatic differentiation.
    #
    # This is the same idea as rinzler-autograd's Value class, but operating
    # on entire matrices at once instead of individual scalars. That's the
    # core reason neural networks are practical — instead of running backprop
    # through millions of individual numbers one at a time, we run it through
    # a handful of matrix operations in batch.
    class Tensor
      attr_accessor :data, :grad
      attr_reader :children, :op

      def initialize(data, children: [], op: nil)
        @data      = coerce_data(data)
        @grad      = Numo::DFloat.zeros(*@data.shape)
        @children  = children
        @op        = op
        @_backward = -> {}
      end

      # ── Shape helpers ─────────────────────────────────────────────────────────

      def shape = @data.shape
      def ndim  = @data.ndim
      def size  = @data.size

      # ── Factory methods ───────────────────────────────────────────────────────

      def self.zeros(*shape)  = new(Numo::DFloat.zeros(*shape))
      def self.ones(*shape)   = new(Numo::DFloat.ones(*shape))
      def self.randn(*shape)  = new(Numo::DFloat.new(*shape).rand_norm)
      def self.rand(*shape)   = new(Numo::DFloat.new(*shape).rand)

      # Convert a nested Ruby array directly: Tensor.from([[1,2],[3,4]])
      def self.from(array) = new(Numo::DFloat.cast(array))

      # ── Arithmetic ────────────────────────────────────────────────────────────

      def +(other)
        other = coerce_tensor(other)
        out   = Tensor.new(@data + other.data, children: [self, other], op: :+)

        # Addition gradient is 1 for both inputs. But if shapes differ due to
        # broadcasting, we must sum the gradient back over the broadcast axes
        # so it matches the original shape.
        out._set_backward do
          self.grad.inplace  + unbroadcast(out.grad, shape)
          other.grad.inplace + unbroadcast(out.grad, other.shape)
        end

        out
      end

      def *(other)
        other = coerce_tensor(other)
        out   = Tensor.new(@data * other.data, children: [self, other], op: :*)

        # Element-wise multiply: d(a*b)/da = b, d(a*b)/db = a.
        # Fast path: when shapes match, one C pass accumulates both grads
        # simultaneously (no temporaries, half the memory traffic).
        # Broadcast path: fall back to Ruby unbroadcast for mismatched shapes.
        out._set_backward do
          if TENSOR_NATIVE && shape == other.shape
            Rinzler::Tensor::TensorExt.mul_backward(@data, other.data, out.grad, self.grad, other.grad)
          else
            self.grad.inplace  + unbroadcast(other.data * out.grad, shape)
            other.grad.inplace + unbroadcast(@data * out.grad, other.shape)
          end
        end

        out
      end

      def -@  = self * -1.0
      def -(other) = self + (-coerce_tensor(other))
      def /(other) = self * (coerce_tensor(other) ** -1)

      def **(exp)
        raise ArgumentError, "exponent must be numeric" if exp.is_a?(Tensor)

        out = Tensor.new(@data ** exp, children: [self], op: :"**#{exp}")
        out._set_backward do
          self.grad.inplace + (exp * @data ** (exp - 1)) * out.grad
        end
        out
      end

      # ── Matrix operations ─────────────────────────────────────────────────────

      # Matrix multiply (forward: C = A · B)
      #
      # Gradient derivation:
      #   dL/dA = dL/dC · Bᵀ    (shape: [m,k] = [m,n] · [n,k]ᵀ)
      #   dL/dB = Aᵀ · dL/dC    (shape: [k,n] = [k,m]ᵀ · [m,n])
      #
      # Intuition: A's gradient asks "how does each element of A affect the
      # loss?" — it's the output gradient dotted with B's contribution.
      def dot(other)
        other    = coerce_tensor(other)
        c_data   = if Rinzler::Tensor.backend == :vulkan
                     Rinzler::Vulkan.matmul(@data, other.data)
                   else
                     mat_dot(@data, other.data)
                   end
        out = Tensor.new(c_data, children: [self, other], op: :dot)

        out._set_backward do
          self.grad.inplace  + mat_dot(out.grad, other.data.transpose)
          other.grad.inplace + mat_dot(@data.transpose, out.grad)
        end

        out
      end

      # Batched matrix multiply: [B, T, d] × [B, d, S] → [B, T, S]
      #
      # Applies independent matrix multiplication for each batch item.
      # This is the fundamental operation for computing QKᵀ across a batch
      # of sequences in multi-head attention — rather than looping over the
      # sequence, we loop over the (much smaller) batch dimension.
      #
      # Gradients follow the same rule as dot(), applied per batch item:
      #   dL/dA[i] = dL/dC[i] · B[i]ᵀ
      #   dL/dB[i] = A[i]ᵀ · dL/dC[i]
      def bmm(other)
        other = coerce_tensor(other)
        b     = shape[0]

        result = if Rinzler::Tensor.backend == :vulkan
                   slices = (0...b).map { |i| Rinzler::Vulkan.matmul(@data[i, true, true], other.data[i, true, true]) }
                   r = Numo::DFloat.zeros(b, slices[0].shape[0], slices[0].shape[1])
                   slices.each_with_index { |s, i| r[i, true, true] = s }
                   r
                 elsif TENSOR_NATIVE
                   Rinzler::Tensor::TensorExt.bmm(@data, other.data)
                 else
                   slices = (0...b).map { |i| mat_dot(@data[i, true, true], other.data[i, true, true]) }
                   r = Numo::DFloat.zeros(b, slices[0].shape[0], slices[0].shape[1])
                   slices.each_with_index { |s, i| r[i, true, true] = s }
                   r
                 end

        out = Tensor.new(result, children: [self, other], op: :bmm)

        out._set_backward do
          if TENSOR_NATIVE
            # dA = dC × Bᵀ,  dB = Aᵀ × dC — same batched kernel, transposed args
            self.grad.inplace  + Rinzler::Tensor::TensorExt.bmm(out.grad, other.data.transpose(0, 2, 1))
            other.grad.inplace + Rinzler::Tensor::TensorExt.bmm(@data.transpose(0, 2, 1), out.grad)
          else
            b.times do |i|
              self.grad[i, true, true]  = self.grad[i, true, true]  +
                mat_dot(out.grad[i, true, true], other.data[i, true, true].transpose)
              other.grad[i, true, true] = other.grad[i, true, true] +
                mat_dot(@data[i, true, true].transpose, out.grad[i, true, true])
            end
          end
        end

        out
      end

      # ── Reductions ────────────────────────────────────────────────────────────

      # Sum all elements or along an axis.
      #
      # Backward: the gradient is broadcast back to every element that
      # contributed to the sum — like pouring water back into every cup
      # that was emptied into the bucket.
      def sum(axis: nil)
        result = axis.nil? ? @data.sum : @data.sum(axis)
        out    = Tensor.new(result, children: [self], op: :sum)

        out._set_backward do
          if axis.nil?
            # out.grad is scalar — numo broadcasts it to fill @data.shape in-place
            self.grad.inplace + out.grad
          else
            # Re-insert the collapsed axis so numo can broadcast back
            expanded_shape       = @data.shape.dup
            expanded_shape[axis] = 1
            self.grad.inplace + out.grad.reshape(*expanded_shape)
          end
        end

        out
      end

      def mean(axis: nil)
        n      = axis.nil? ? @data.size : @data.shape[axis]
        result = axis.nil? ? @data.mean : @data.mean(axis)
        out    = Tensor.new(result, children: [self], op: :mean)

        out._set_backward do
          if axis.nil?
            self.grad.inplace + (out.grad / n)
          else
            expanded_shape       = @data.shape.dup
            expanded_shape[axis] = 1
            self.grad.inplace + (out.grad.reshape(*expanded_shape) / n)
          end
        end

        out
      end

      # ── Shape operations ──────────────────────────────────────────────────────

      # Slice `length` columns starting at `start` along the last axis.
      # Works for any ndim: 2D [T, C] or 3D [B, T, C].
      # Used to split a combined QKV projection into Q, K, V.
      #
      # Backward: scatter the incoming gradient back into the right column range.
      def slice_cols(start, length)
        last = ndim - 1
        idx  = Array.new(ndim, 0..-1)
        idx[last] = start...(start + length)

        out = Tensor.new(@data[*idx], children: [self], op: :slice_cols)

        out._set_backward do
          self.grad[*idx] = self.grad[*idx] + out.grad
        end

        out
      end

      # Concatenate a list of tensors along their last axis.
      # Works for any ndim: 2D [T, C] or 3D [B, T, C].
      # Used to rejoin attention heads after computing them independently.
      #
      # Backward: split the upstream gradient back to each input's column range.
      def self.concat_cols(tensors)
        raise ArgumentError, "concat_cols requires at least one tensor" if tensors.empty?

        n    = tensors.first.ndim
        last = n - 1

        data = if n == 2
          Numo::DFloat.hstack(tensors.map(&:data))
        else
          # For 3D+: flatten leading dims, hstack, reshape back
          leading = tensors.first.shape[0...-1]
          flat    = tensors.map { |t| t.data.reshape(leading.reduce(:*), t.shape[-1]) }
          stacked = Numo::DFloat.hstack(flat)
          total_c = tensors.sum { |t| t.shape[-1] }
          stacked.reshape(*leading, total_c)
        end

        out = new(data, children: tensors, op: :concat_cols)

        out._set_backward do
          offset = 0
          tensors.each do |t|
            w   = t.shape[last]
            idx = Array.new(n, 0..-1)
            idx[last] = offset...(offset + w)
            t.grad.inplace + out.grad[*idx]
            offset += w
          end
        end

        out
      end

      def reshape(*new_shape)
        original_shape = shape
        out = Tensor.new(@data.reshape(*new_shape), children: [self], op: :reshape)

        out._set_backward do
          self.grad.inplace + out.grad.reshape(*original_shape)
        end

        out
      end

      def transpose
        out = Tensor.new(@data.transpose, children: [self], op: :transpose)
        out._set_backward { self.grad.inplace + out.grad.transpose }
        out
      end

      alias_method :T, :transpose

      # Transpose the last two dimensions: [B, T, d] → [B, d, T].
      # Used in attention for computing QKᵀ across a batch.
      # This operation is its own inverse, so the backward uses the same axis swap.
      def transpose_last2
        axes       = (0...ndim).to_a
        axes[-2], axes[-1] = axes[-1], axes[-2]

        out = Tensor.new(@data.transpose(*axes), children: [self], op: :transpose_last2)
        out._set_backward { self.grad.inplace + out.grad.transpose(*axes) }
        out
      end

      # ── Activations ───────────────────────────────────────────────────────────

      def relu
        out = Tensor.new(@data.clip(0, Float::INFINITY), children: [self], op: :relu)

        # Gradient is 1 where the input was positive, 0 where it was clamped.
        out._set_backward do
          mask      = Numo::DFloat.cast(@data > 0)
          self.grad = self.grad + mask * out.grad
        end

        out
      end

      def tanh
        t   = Numo::NMath.tanh(@data)
        out = Tensor.new(t, children: [self], op: :tanh)

        out._set_backward do
          self.grad = self.grad + (1.0 - t ** 2) * out.grad
        end

        out
      end

      def exp
        e   = Numo::NMath.exp(@data)
        out = Tensor.new(e, children: [self], op: :exp)

        # e^x is its own derivative.
        out._set_backward { self.grad = self.grad + e * out.grad }

        out
      end

      def log
        out = Tensor.new(Numo::NMath.log(@data), children: [self], op: :log)

        out._set_backward do
          self.grad = self.grad + (1.0 / @data) * out.grad
        end

        out
      end

      # Numerically stable log-softmax along the last axis.
      #
      # More efficient than log(softmax(x)) — avoids exp → log round-trip and
      # the underflow that comes with it for large negative logits.
      #
      # Forward:  y = x - max(x) - log(Σ exp(x - max(x)))
      # Backward: dL/dx = dy - softmax(x) · Σ dy
      #
      # Used for cross-entropy loss: nll = -log_softmax(logits)[target_id]
      # This collapses to a gather from y, which we implement via one-hot multiply.
      def log_softmax
        last_axis    = ndim - 1
        batch_shape  = shape[0...-1]
        expand_shape = [*batch_shape, 1]

        row_max   = @data.max(last_axis).reshape(*expand_shape)
        shifted   = @data - row_max
        log_sum   = Numo::NMath.log(Numo::NMath.exp(shifted).sum(last_axis)).reshape(*expand_shape)
        y         = shifted - log_sum             # log-probabilities
        softmax_x = Numo::NMath.exp(y)            # = softmax(x), reused in backward

        out = Tensor.new(y, children: [self], op: :log_softmax)

        out._set_backward do
          sum_dy    = out.grad.sum(last_axis).reshape(*expand_shape)
          self.grad = self.grad + out.grad - softmax_x * sum_dy
        end

        out
      end

      # Numerically stable softmax along the last axis.
      #
      # Works for any ndim: 2D [batch, features] or 3D [batch, seq, features].
      #
      # We subtract the row maximum before exponentiating — this prevents
      # overflow without changing the result (e^(x-c)/Σe^(x-c) = e^x/Σe^x).
      #
      # Backward: the Jacobian of softmax is s·(δᵢⱼ - sⱼ), which simplifies
      # to: grad_input = s * (grad_output - (grad_output * s).sum(last_axis, keepdims))
      def softmax
        last_axis    = ndim - 1
        batch_shape  = shape[0...-1]
        expand_shape = [*batch_shape, 1]

        row_max  = @data.max(last_axis).reshape(*expand_shape)
        shifted  = @data - row_max
        e        = Numo::NMath.exp(shifted)
        row_sums = e.sum(last_axis).reshape(*expand_shape)
        s        = e / row_sums

        out = Tensor.new(s, children: [self], op: :softmax)

        out._set_backward do
          dot_gs    = (out.grad * s).sum(last_axis).reshape(*expand_shape)
          self.grad = self.grad + s * (out.grad - dot_gs)
        end

        out
      end

      # ── Fused cross-entropy loss ──────────────────────────────────────────────

      # Tensor.cross_entropy(logits, targets) → scalar Tensor
      #
      # logits:  Tensor [n, vocab_size]   (float logits, not probabilities)
      # targets: Array of Integer         (correct class index per row, length n)
      #
      # Returns: scalar Tensor — mean NLL loss over all n rows.
      #
      # Why fused?
      #   The naive chain is: log_softmax → (* one_hot) → sum(axis:1) → mean.
      #   That's 5 backward nodes and allocates a full [n, vocab_size] one_hot
      #   matrix (~10 MB at batch=8, context=128, vocab=1250).
      #
      #   This fuses everything into one node. The backward computes
      #   d(CE)/d(logits) = (softmax(logits) - one_hot) / n  entirely in-place
      #   via numo fancy indexing — no one_hot array, one backward node.
      #
      # The log_softmax / softmax duality on the integer-indexed path:
      #   Forward uses log-space for numerical stability (no exp-then-log).
      #   Backward uses softmax = exp(log_prob) which is already computed
      #   during forward, so it's free to reuse.
      def self.cross_entropy(logits, targets)
        n     = logits.shape[0]
        vocab = logits.shape[1]

        # Numerically stable log-softmax
        row_max  = logits.data.max(1).reshape(n, 1)
        shifted  = logits.data - row_max
        exp_s    = Numo::NMath.exp(shifted)
        sum_exp  = exp_s.sum(1).reshape(n, 1)
        log_prob = shifted - Numo::NMath.log(sum_exp)   # [n, vocab]
        softmax  = exp_s / sum_exp                       # kept for backward

        # Gather correct-class log-probs. Numo's [int_arr, int_arr] indexing does a
        # cross-product (outer select), not element-wise — use flat indices instead.
        flat_idx = Numo::Int32.cast(targets.each_with_index.map { |t, i| i * vocab + t })
        flat_log = log_prob.reshape(n * vocab)
        nll_val  = -flat_log[flat_idx].mean

        out = Tensor.new(Numo::DFloat[nll_val], children: [logits], op: :cross_entropy)

        out._set_backward do
          # d(CE)/d(logits[i,j]) = softmax[i,j]/n        (j != target[i])
          #                       = (softmax[i,j] - 1)/n  (j == target[i])
          # One [n, vocab] allocation; single backward node.
          # Note: numo reshape returns a copy, not a view — subtract at targets
          # directly on the 2D array using Ruby iteration (O(n), n=B*T).
          inv_n = 1.0 / n
          grad  = softmax * inv_n
          targets.each_with_index { |tid, i| grad[i, tid] -= inv_n }
          logits.grad.inplace + grad
        end

        out
      end

      # ── Backpropagation ───────────────────────────────────────────────────────

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

        # Seed: gradient of the output with respect to itself is all ones.
        @grad = Numo::DFloat.ones(*@data.shape)

        topo.reverse_each(&:_backward)
      end

      # ── Graph management ─────────────────────────────────────────────────────

      # Release the computation graph after backward is complete.
      #
      # Walks all nodes reachable from self and nils out their backward procs
      # and children references. This allows Ruby's GC to collect every
      # intermediate tensor immediately rather than waiting for the next
      # training step to overwrite the `loss` variable.
      #
      # Only `data` and `grad` survive — which is all the optimizer needs.
      # Call this after loss.backward and before reading loss.data.
      NOOP           = -> {}
      EMPTY_CHILDREN = [].freeze

      def free_graph!
        visited = {}
        stack   = [self]
        until stack.empty?
          node = stack.pop
          next if visited[node.object_id]
          visited[node.object_id] = true
          stack.concat(node.children)
          node.instance_variable_set(:@children,  EMPTY_CHILDREN)
          node.instance_variable_set(:@_backward, NOOP)
        end
        self
      end

      # ── Support ───────────────────────────────────────────────────────────────

      def coerce(other) = [Tensor.new(other), self]

      def to_s    = "Tensor(shape=#{shape.inspect}, op=#{op.inspect})"
      def inspect = to_s

      def _set_backward(&) = @_backward = Proc.new(&)
      def _backward        = @_backward.call

      private

      # Accept Numo arrays, nested Ruby arrays, scalars, or other Tensors.
      # Route matrix multiply through BLAS when the matrix is large enough
      # to overcome the call overhead, and numo-linalg is available.
      def mat_dot(a, b)
        if LINALG_AVAILABLE && a.shape.any? { |d| d >= LINALG_THRESHOLD }
          Numo::Linalg.dot(a, b)
        else
          a.dot(b)
        end
      end

      def coerce_data(data)
        case data
        when Numo::DFloat  then data
        when Numo::NArray  then data.cast_to(Numo::DFloat)
        when Array         then Numo::DFloat.cast(data)
        when Numeric       then Numo::DFloat[data]
        when Tensor        then data.data
        else raise ArgumentError, "cannot build Tensor from #{data.class}"
        end
      end

      def coerce_tensor(other) = other.is_a?(Tensor) ? other : Tensor.new(other)

      # When a tensor was broadcast during a forward op, its gradient arrives
      # with the larger broadcast shape. We must sum it back down to the
      # original shape before accumulating.
      #
      # Example: a[3] + b[2,3] → out[2,3]
      #   b's grad is fine (shape matches out).
      #   a's grad must be summed over axis 0: [2,3] → [3].
      def unbroadcast(grad, target_shape)
        return grad if grad.shape == target_shape

        result = grad

        # If grad has more dimensions, sum away the leading ones.
        while result.ndim > target_shape.size
          result = result.sum(0)
        end

        # Sum over any axis where the target was size 1 (it was broadcast).
        target_shape.each_with_index do |dim, axis|
          next unless dim == 1 && result.shape[axis] != 1

          summed = result.sum(axis)
          # numo returns a plain Ruby Float when summing a 1D array — coerce back.
          summed = Numo::DFloat[summed] unless summed.is_a?(Numo::NArray)
          # Re-insert the size-1 dimension that was collapsed.
          result = summed.reshape(*result.shape.first(axis), 1, *result.shape.drop(axis + 1))
        end

        result
      end
    end
  end
end
