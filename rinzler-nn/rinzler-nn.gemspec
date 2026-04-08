# frozen_string_literal: true

require_relative "lib/rinzler/nn/version"

Gem::Specification.new do |spec|
  spec.name = "rinzler-nn"
  spec.version = Rinzler::NN::VERSION
  spec.authors = ["Matrix9180"]
  spec.email = ["matrix9180@proton.me"]

  spec.summary = "Neural network layers for the Rinzler ML framework."
  spec.description = "Module, Parameter, Linear, Embedding, LayerNorm — the building blocks for transformers, built on rinzler-tensor."
  spec.homepage = "https://github.com/matrix9180/rinzler"
  spec.required_ruby_version = ">= 4.0.0"
  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  gemspec = File.basename(__FILE__)
  spec.files = IO.popen(%w[git ls-files -z], chdir: __dir__, err: IO::NULL) do |ls|
    ls.readlines("\x0", chomp: true).reject do |f|
      (f == gemspec) ||
        f.start_with?(*%w[bin/ Gemfile .gitignore])
    end
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_dependency "rinzler-tensor", "~> 0.1"

  spec.add_development_dependency "minitest", "~> 5.0"

  # For more information and examples about making a new gem, check out our
  # guide at: https://bundler.io/guides/creating_gem.html
end
