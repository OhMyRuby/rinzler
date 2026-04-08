# frozen_string_literal: true

require_relative "lib/rinzler/vulkan/version"

Gem::Specification.new do |spec|
  spec.name    = "rinzler-vulkan"
  spec.version = Rinzler::Vulkan::VERSION
  spec.authors = ["Matrix9180"]
  spec.email   = ["matrix9180@proton.me"]

  spec.summary  = "Vulkan GPU backend for Rinzler tensor operations"
  spec.homepage = "https://github.com/matrix9180/rinzler"
  spec.required_ruby_version = ">= 3.2.0"

  spec.files = Dir[
    "lib/**/*.rb",
    "ext/**/*.{rb,c,h}",
    "shaders/**/*.{comp,spv}",
  ]

  spec.require_paths = ["lib"]
  spec.extensions    = ["ext/rinzler/vulkan/extconf.rb"]
end
