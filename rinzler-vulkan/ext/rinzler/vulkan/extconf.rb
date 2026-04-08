require "mkmf"

# Vulkan headers
unless find_header("vulkan/vulkan.h")
  abort "vulkan/vulkan.h not found — install vulkan-headers (pacman -S vulkan-headers)"
end

# Vulkan loader
unless find_library("vulkan", "vkCreateInstance")
  abort "libvulkan not found — install vulkan-icd-loader"
end

# Compile the GLSL shader to SPIR-V at build time.
# glslc ships with the shaderc package.
shader_src = File.expand_path("../../../shaders/gemm.comp", __dir__)
shader_spv = File.expand_path("../../../shaders/gemm.spv", __dir__)

unless system("glslc --target-env=vulkan1.2 #{shader_src} -o #{shader_spv}")
  abort "Shader compilation failed — is glslc installed? (pacman -S shaderc)"
end

puts "Shader compiled: #{shader_spv}"

$CFLAGS  += " -std=c11 -Wall -O2"
$LDFLAGS += " -lvulkan"

create_makefile("rinzler/vulkan/vulkan_ext")
