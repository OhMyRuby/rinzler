# frozen_string_literal: true

require "rinzler/tensor"
require_relative "optim/version"
require_relative "optim/optimizer"
require_relative "optim/sgd"
require_relative "optim/sgd_momentum"
require_relative "optim/rmsprop"
require_relative "optim/adam"
require_relative "optim/adam_w"
require_relative "optim/scheduler"
require_relative "optim/linear_warmup"
require_relative "optim/cosine_with_warmup"

module Rinzler
  module Optim
    class Error < StandardError; end
  end
end
