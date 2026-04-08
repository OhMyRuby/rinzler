# frozen_string_literal: true

module Rinzler
  module NN
    # Embedding is a lookup table: integer token IDs → dense vectors.
    #
    # Conceptually it's a matrix of shape [vocab_size, embedding_dim].
    # The forward pass selects rows by index. The backward pass scatters
    # gradients back to whichever rows were selected.
    #
    # This is how every language model turns token IDs into something
    # the network can do math with. The embedding weights are learned
    # just like any other parameter.
    #
    # Example:
    #   emb = Embedding.new(50257, 768)   # GPT-2 vocab, 768-dim embeddings
    #   out = emb.call([1, 42, 7])        # → Tensor [3, 768]
    class Embedding < Module
      attr_reader :weight

      def initialize(num_embeddings, embedding_dim)
        @num_embeddings = num_embeddings
        @embedding_dim  = embedding_dim

        # Small normal init — standard for embeddings
        @weight = Parameter.new(Numo::DFloat.new(num_embeddings, embedding_dim).rand_norm * 0.02)
      end

      # indices: 1D Array of Integer token IDs → Tensor [seq_len, embedding_dim]
      #          2D Array [[...],[...]]         → Tensor [batch, seq_len, embedding_dim]
      def forward(indices)
        # Detect batch dimension
        batched     = indices.first.is_a?(Array)
        batch_shape = batched ? [indices.size, indices[0].size] : [Array(indices).size]
        flat        = Array(indices).flatten
        w           = @weight

        out_data = Numo::DFloat.zeros(flat.size, @embedding_dim)
        flat.each_with_index { |idx, i| out_data[i, true] = w.data[idx, true] }

        # Reshape to [seq_len, dim] or [batch, seq_len, dim]
        out_data = out_data.reshape(*batch_shape, @embedding_dim)

        out = Rinzler::Tensor::Tensor.new(out_data, children: [w], op: :embedding)

        # Backward: scatter incoming gradients back to the rows of @weight
        # that were selected. If the same token appears multiple times,
        # their gradients accumulate (+=) — correct behaviour.
        out._set_backward do
          flat_grad = out.grad.reshape(flat.size, @embedding_dim)
          flat.each_with_index do |idx, i|
            w.grad[idx, true] = w.grad[idx, true] + flat_grad[i, true]
          end
        end

        out
      end
    end
  end
end
