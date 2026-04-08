require "mkmf"

$CFLAGS += " -std=c11 -Wall -O3 -march=native"

# Locate numo-narray headers. extconf runs without bundler, so we search the
# gem paths directly for a directory containing numo/narray.h.
gem_paths = Gem.paths.home ? [Gem.paths.home] : []
gem_paths += Gem.path

narray_inc = gem_paths
  .flat_map { |gp| Dir.glob(File.join(gp, "{gems,bundler/gems}", "numo-narray-*", "ext/numo/narray")) }
  .find { |d| File.exist?(File.join(d, "numo/narray.h")) }

abort "Could not locate numo-narray headers. Is numo-narray installed?" unless narray_inc
$INCFLAGS << " -I#{narray_inc}"

# na_data_type and other numo symbols are resolved at runtime from the already-loaded
# narray.so — no link-time dependency needed. Ruby guarantees numo/narray is loaded
# before this extension because tensor.rb requires numo/narray first.

create_makefile("rinzler/tensor/tensor_ext")
