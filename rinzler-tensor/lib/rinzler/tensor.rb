# frozen_string_literal: true

require_relative "../rinzler/logger"
require_relative "tensor/version"
require_relative "tensor/tensor"

# Load native extension after numo/narray is already in place — tensor_ext.so
# resolves na_data_type and friends from the already-loaded narray.so at runtime.
begin
  require "rinzler/tensor/tensor_ext"
  TENSOR_NATIVE = true
rescue LoadError
  TENSOR_NATIVE = false
end

module Rinzler
  module Tensor
    class Error < StandardError; end
  end
end
