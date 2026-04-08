# frozen_string_literal: true

require_relative "../rinzler/logger"
require_relative "tensor/version"
require_relative "tensor/tensor"

module Rinzler
  module Tensor
    class Error < StandardError; end
  end
end
