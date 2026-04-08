# profile_step.rb — profile a handful of training steps to find the bottleneck.
#
# Uses the same model/data setup as train.rb but skips CLI parsing and
# runs only PROFILE_STEPS steps inside a StackProf wall-clock window.
#
# Usage:
#   bundle exec ruby rinzler-gpt/profile_step.rb
#
# Output: rinzler-gpt/profile.dump  (load with stackprof --text)
#   bundle exec stackprof rinzler-gpt/profile.dump --text --limit 30

require_relative "lib/rinzler/gpt"
require "json"

PROFILE_STEPS = 20
CONTEXT       = 128
BATCH_SIZE    = 8
D_MODEL       = 64
N_HEADS       = 4
N_LAYERS      = 4
VOCAB_SIZE    = 1250  # merges — adjust to match your current tokenizer cache
OUT_FILE      = File.join(__dir__, "profile.dump")
TOKENIZER_CACHE = Dir.glob(File.join(__dir__, "tokenizer_#{VOCAB_SIZE}.json")).first

abort "No tokenizer cache found for vocab_size=#{VOCAB_SIZE}. Run train.rb first." unless TOKENIZER_CACHE

puts "Loading tokenizer from #{TOKENIZER_CACHE}..."
tokenizer = Rinzler::Tokenizer::BPE.from_file(TOKENIZER_CACHE)
puts "  #{tokenizer.vocab_size} tokens."

ids_cache = TOKENIZER_CACHE.sub(".json", "_ids.bin")
if File.exist?(ids_cache)
  all_ids = File.binread(ids_cache).unpack("l<*")
  puts "  #{all_ids.size} token IDs loaded from cache."
else
  abort "No IDs cache found at #{ids_cache}. Run train.rb once to build it."
end

split     = (all_ids.size * 0.9).to_i
train_ids = all_ids[0...split]

def random_batch(ids, context_len, batch_size)
  batch_size.times.map do
    start = rand(ids.size - context_len - 1)
    ids[start...(start + context_len + 1)]
  end
end

cfg = Rinzler::GPT::Config.new(
  vocab_size:  tokenizer.vocab_size,
  context_len: CONTEXT,
  d_model:     D_MODEL,
  n_heads:     N_HEADS,
  n_layers:    N_LAYERS,
  ffn_mult:    4
)
model = Rinzler::GPT::GPTModel.new(cfg)
opt   = Rinzler::Optim::AdamW.new(model.parameters, lr: 3e-4, weight_decay: 0.1)

# Warm up outside the profiling window so JIT/GC settle
3.times do
  batch = random_batch(train_ids, CONTEXT, BATCH_SIZE)
  opt.zero_grad
  loss = model.loss(batch)
  loss.backward
  loss.free_graph!
  opt.step
end

puts "Warm-up done. Profiling #{PROFILE_STEPS} steps..."

def t = Process.clock_gettime(Process::CLOCK_MONOTONIC)

# ── Phase 1: coarse stage breakdown ──────────────────────────────────────────

stage_times  = Hash.new(0.0)

PROFILE_STEPS.times do
  batch = random_batch(train_ids, CONTEXT, BATCH_SIZE)
  t0 = t; opt.zero_grad;           stage_times[:zero_grad]  += t - t0
  t0 = t; loss = model.loss(batch); stage_times[:forward]    += t - t0
  t0 = t; loss.backward;           stage_times[:backward]   += t - t0
  t0 = t; loss.free_graph!;        stage_times[:free_graph] += t - t0
  t0 = t; opt.step;                stage_times[:opt_step]   += t - t0
end

total = stage_times.values.sum
puts "\n%-14s %8s %8s %6s" % %w[stage total(s) per_step %total]
puts "-" * 42
stage_times.each { |k,v| puts "%-14s %8.3f %8.3f %5.1f%%" % [k, v, v/PROFILE_STEPS, v/total*100] }
puts "-" * 42
puts "%-14s %8.3f %8.3f" % ["TOTAL", total, total/PROFILE_STEPS]

# ── Phase 2: per-op breakdown inside backward ─────────────────────────────────
#
# Patch Tensor#backward to time each node's _backward call by op type.
# We record into op_times which accumulates across all steps.

$op_times  = Hash.new(0.0)
$op_counts = Hash.new(0)

Rinzler::Tensor::Tensor.prepend(Module.new do
  def backward
    topo    = []
    visited = Set.new
    build_topo = ->(node) {
      next if visited.include?(node)
      visited.add(node)
      node.children.each { build_topo.call(it) }
      topo << node
    }
    build_topo.call(self)
    @grad = Numo::DFloat.ones(*@data.shape)
    topo.reverse_each do |node|
      key = node.op || :leaf
      t0  = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      node._backward
      $op_times[key]  += Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0
      $op_counts[key] += 1
    end
  end
end)

PROFILE_STEPS.times do
  batch = random_batch(train_ids, CONTEXT, BATCH_SIZE)
  opt.zero_grad
  loss = model.loss(batch)
  loss.backward
  loss.free_graph!
  opt.step
end

back_total = $op_times.values.sum
puts "\n── backward op breakdown (#{PROFILE_STEPS} steps) ──"
puts "%-20s %8s %8s %8s %6s" % %w[op total(s) per_step calls %back]
puts "-" * 58
$op_times.sort_by { |_,v| -v }.each do |op, secs|
  calls = $op_counts[op]
  puts "%-20s %8.3f %8.4f %8d %5.1f%%" % [op, secs, secs/PROFILE_STEPS, calls, secs/back_total*100]
end
puts "-" * 58
puts "%-20s %8.3f %8.3f" % ["TOTAL_BACKWARD", back_total, back_total/PROFILE_STEPS]
