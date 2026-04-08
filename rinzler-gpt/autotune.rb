# frozen_string_literal: true

# autotune.rb — benchmark batch_size × OMP_NUM_THREADS to find the fastest config.
#
# Runs a short fixed-step training loop for each combination and measures
# steps/sec. Prints a ranked table and emits the optimal flags for train.rb.
#
# Usage:
#   bundle exec ruby autotune.rb
#   bundle exec ruby autotune.rb --vulkan
#   bundle exec ruby autotune.rb --batch-sizes "4,8,16,32" --threads "1,2,4,8"
#   bundle exec ruby autotune.rb --steps 30   # steps per candidate (default 20)

require_relative "lib/rinzler/gpt"
require "optparse"

$stdout.sync = true

BENCH_VOCAB   = 256   # small fixed vocab — we're measuring throughput, not quality
BENCH_CONTEXT = 64
BENCH_D_MODEL = 64
BENCH_LAYERS  = 2
BENCH_HEADS   = 4

options = {
  steps:       20,
  batch_sizes: [4, 8, 16, 32],
  threads:     [1, 2, 4],
  vulkan:      false
}

OptionParser.new do |o|
  o.on("--steps N",        Integer) { |v| options[:steps]       = v }
  o.on("--batch-sizes CSV")         { |v| options[:batch_sizes] = v.split(",").map(&:to_i) }
  o.on("--threads CSV")             { |v| options[:threads]     = v.split(",").map(&:to_i) }
  o.on("--vulkan")                  { |_| options[:vulkan]      = true }
end.parse!

if options[:vulkan]
  require "rinzler/vulkan"
  Rinzler::Tensor.backend = :vulkan
  puts "GPU backend: Vulkan"
end

puts "Autotune — #{options[:batch_sizes].size * options[:threads].size} candidates, #{options[:steps]} steps each.\n\n"

# Tiny synthetic corpus — random token IDs, no disk I/O in the benchmark loop.
corpus_ids = Array.new(50_000) { rand(BENCH_VOCAB) }

def random_batch(ids, context_len, batch_size)
  batch_size.times.map do
    start = rand(ids.size - context_len - 1)
    ids[start...(start + context_len + 1)]
  end
end

results = []

options[:threads].each do |threads|
  ENV["OMP_NUM_THREADS"] = threads.to_s

  options[:batch_sizes].each do |batch_size|
    cfg   = Rinzler::GPT::Config.new(
      vocab_size:  BENCH_VOCAB,
      context_len: BENCH_CONTEXT,
      d_model:     BENCH_D_MODEL,
      n_heads:     BENCH_HEADS,
      n_layers:    BENCH_LAYERS,
      ffn_mult:    4
    )
    model = Rinzler::GPT::GPTModel.new(cfg)
    opt   = Rinzler::Optim::AdamW.new(model.parameters, lr: 3e-4, weight_decay: 0.1)

    # Warmup pass — JIT, memory allocation, first-call overhead.
    batch = random_batch(corpus_ids, BENCH_CONTEXT, batch_size)
    opt.zero_grad
    model.loss(batch).backward
    opt.step

    t0 = Time.now

    options[:steps].times do
      batch = random_batch(corpus_ids, BENCH_CONTEXT, batch_size)
      opt.zero_grad
      model.loss(batch).backward
      opt.step
    end

    elapsed    = Time.now - t0
    steps_sec  = (options[:steps] / elapsed).round(2)
    tokens_sec = (options[:steps] * batch_size * BENCH_CONTEXT / elapsed).round(0)

    results << { threads:, batch_size:, steps_sec:, tokens_sec: }
    puts "  threads=#{threads.to_s.rjust(2)}  batch=#{batch_size.to_s.rjust(2)}  →  #{steps_sec.to_s.rjust(7)} steps/s  #{tokens_sec.to_s.rjust(8)} tokens/s"
  end
end

best = results.max_by { |r| r[:tokens_sec] }

puts "\n#{"─" * 60}"
puts "Optimal config:  --batch-size #{best[:batch_size]}"
puts "                 OMP_NUM_THREADS=#{best[:threads]}"
puts "                 #{best[:steps_sec]} steps/s  #{best[:tokens_sec]} tokens/s"
puts "\nRun:"
vulkan_flag = options[:vulkan] ? " --vulkan" : ""
puts "  OMP_NUM_THREADS=#{best[:threads]} bundle exec ruby train.rb --batch-size #{best[:batch_size]}#{vulkan_flag} [...]"
