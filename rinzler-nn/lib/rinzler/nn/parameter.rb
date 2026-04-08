# frozen_string_literal: true

module Rinzler
  module NN
    # Parameter is a Tensor that's part of a model's learnable state.
    #
    # Functionally it's identical to Tensor — the distinction is semantic.
    # Module#parameters walks a module's instance variables looking for
    # Parameter instances so the optimizer knows what to update.
    #
    # This mirrors PyTorch's nn.Parameter: a tensor that "opts in" to
    # gradient tracking and automatic discovery by the module system.
    class Parameter < Rinzler::Tensor::Tensor
    end
  end
end
