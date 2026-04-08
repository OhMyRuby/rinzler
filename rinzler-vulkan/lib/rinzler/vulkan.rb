# frozen_string_literal: true

require_relative "vulkan/version"

# Load the compiled C extension. The extension registers:
#   Rinzler::Vulkan.init(shader_spv_path)
#   Rinzler::Vulkan.initialized?
#   Rinzler::Vulkan.gemm(a_flat, b_flat, m, k, n) -> c_flat
require_relative "vulkan/vulkan_ext"

module Rinzler
  module Vulkan
    class Error < StandardError; end

    # Path to the SPIR-V shader compiled at gem build time.
    SHADER_PATH = File.expand_path(
      "../../../shaders/gemm.spv", __FILE__
    ).freeze

    # Auto-initialise on first use.  Idempotent.
    def self.ensure_initialized!
      return if initialized?

      raise Error, "Shader not found at #{SHADER_PATH}. Rebuild the gem." \
        unless File.exist?(SHADER_PATH)

      init(SHADER_PATH)
      Rinzler.logger.info("Vulkan compute backend initialised (#{SHADER_PATH})")
    end

    # High-level matmul: accepts Numo::DFloat matrices, returns Numo::DFloat.
    # Works for 2-D inputs [M,K] × [K,N] → [M,N].
    def self.matmul(a, b)
      ensure_initialized!

      m, k  = a.shape
      k2, n = b.shape
      raise Error, "Shape mismatch: #{k} != #{k2}" unless k == k2

      a_flat = a.flatten.to_a
      b_flat = b.flatten.to_a

      c_flat = gemm(a_flat, b_flat, m, k, n)

      Numo::DFloat.cast(c_flat).reshape(m, n)
    end
  end
end
