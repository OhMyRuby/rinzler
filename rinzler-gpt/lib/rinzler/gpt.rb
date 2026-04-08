# frozen_string_literal: true

require "rinzler/nn"
require "rinzler/optim"
require "rinzler/tokenizer"
require_relative "gpt/version"
require_relative "gpt/attention"
require_relative "gpt/transformer_block"
require_relative "gpt/model"

module Rinzler
  module GPT
    class Error < StandardError; end
  end
end
