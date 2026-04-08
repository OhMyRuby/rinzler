# frozen_string_literal: true

require "rinzler/tensor"
require_relative "nn/version"
require_relative "nn/parameter"
require_relative "nn/module"
require_relative "nn/linear"
require_relative "nn/embedding"
require_relative "nn/layer_norm"

module Rinzler
  module NN
    class Error < StandardError; end
  end
end
