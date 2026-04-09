# frozen_string_literal: true

source "https://rubygems.org"

# Register all sub-gems so Bundler resolves inter-gem dependencies without path: references.
gemspec path: "rinzler-autograd"
gemspec path: "rinzler-tensor"
gemspec path: "rinzler-nn"
gemspec path: "rinzler-optim"
gemspec path: "rinzler-tokenizer"
gemspec path: "rinzler-gpt"
gemspec path: "rinzler-vulkan"

# Patched fork — fixes function pointer type mismatch for Ruby 4 / strict C compilers.
# Pinned until ruby-numo/numo-narray#246 merges upstream.
gem "numo-narray", git: "https://github.com/matrix9180/numo-narray.git"

gem "numo-linalg"
gem "fiddle"
gem "irb"
gem "rake",          "~> 13.0"
gem "rake-compiler"
gem "minitest",      "~> 5.0"
gem "stackprof"
gem "ruby-prof"
