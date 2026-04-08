# frozen_string_literal: true

# train.rb — train a GPT on Why's Poignant Guide to Ruby
#
# Usage:
#   bundle exec ruby train.rb
#   bundle exec ruby train.rb --steps 2000 --lr 3e-4 --batch-size 8
#
# The model learns to predict the next character at every position.
# After enough steps, it generates text that sounds like _why.

require_relative "lib/rinzler/gpt"
require "optparse"
require "fileutils"

$stdout.sync = true  # flush immediately when piped

# ── Config ─────────────────────────────────────────────────────────────────────

options = {
  corpus:     [File.join(__dir__, "corpus.txt")],
  steps:      1000,
  lr:         3e-4,
  batch_size: 8,
  eval_every: 100,
  gen_every:  200,
  gen_len:    200,
  context:    128,
  d_model:    64,
  n_heads:    4,
  n_layers:   4,
  save_every: 500,
  out:        nil,   # resolved after option parsing — nil means "auto"
  resume:     nil,
  vocab_size: 500,   # number of BPE merges; total vocab ≈ unique_chars + vocab_size
  div_window:    5,     # number of eval steps to look back when computing Δgap trend
  div_warn:      20,    # warn when gap% exceeds this value (20 = 20%)
  div_crit:      nil,   # abort training when gap% exceeds this value (50 = 50%; nil = disabled)
  warmup_steps:  0,     # linear LR warmup steps (0 = disabled)
  clip_grad:     1.0    # gradient clipping max norm (nil = disabled)
}

OptionParser.new do |o|
  o.on("--steps N",      Integer) { |v| options[:steps]      = v }
  o.on("--lr N",         Float)   { |v| options[:lr]         = v }
  o.on("--d-model N",    Integer) { |v| options[:d_model]    = v }
  o.on("--layers N",     Integer) { |v| options[:n_layers]   = v }
  o.on("--context N",    Integer) { |v| options[:context]    = v }
  o.on("--batch-size N", Integer) { |v| options[:batch_size] = v }
  o.on("--eval-every N", Integer) { |v| options[:eval_every] = v }
  o.on("--gen-every N",  Integer) { |v| options[:gen_every]  = v }
  o.on("--gen-len N",    Integer) { |v| options[:gen_len]    = v }
  o.on("--save-every N", Integer) { |v| options[:save_every] = v }
  o.on("--vocab-size N", Integer) { |v| options[:vocab_size] = v }
  o.on("--div-window N", Integer) { |v| options[:div_window] = v }
  o.on("--div-warn N",    Float)   { |v| options[:div_warn]      = v }
  o.on("--div-crit N",    Float)   { |v| options[:div_crit]      = v }
  o.on("--warmup-steps N", Integer) { |v| options[:warmup_steps] = v }
  o.on("--clip-grad N",   Float)   { |v| options[:clip_grad]     = v }
  o.on("--no-clip-grad")           { |_| options[:clip_grad]     = nil }
  o.on("--out PATH")              { |v| options[:out]        = v }
  o.on("--resume PATH")           { |v| options[:resume]          = v }
  o.on("--retrain-tokenizer")     { |_| options[:retrain_tokenizer] = true }
  o.on("--vulkan")                { |_| options[:vulkan]            = true }
  o.on("--corpus PATTERN") do |v|
    # First explicit --corpus clears the default; subsequent ones append.
    options[:corpus] = [] if options[:corpus] == [File.join(__dir__, "corpus.txt")]
    options[:corpus] << v
  end
end.parse!

# ── Run directory ──────────────────────────────────────────────────────────────
#
# Fresh runs auto-create runs/<n>/ (next available integer).
# Resumed runs default to the directory of the source checkpoint.
# --out always wins if explicitly provided.

if options[:out].nil?
  if options[:resume]
    options[:out] = File.join(File.dirname(File.expand_path(options[:resume])), "checkpoint.json")
  else
    runs_dir = File.join(__dir__, "runs")
    n = (Dir.glob(File.join(runs_dir, "*/")).map { |d| File.basename(d).to_i }.max || 0) + 1
    run_dir  = File.join(runs_dir, n.to_s)
    FileUtils.mkdir_p(run_dir)
    options[:out] = File.join(run_dir, "checkpoint.json")
    puts "Run directory: #{run_dir}"
  end
end

# Tokenizer cache is project-level, shared across runs — keyed by vocab size.
# Rebuilding BPE is expensive and the result is deterministic for the same corpus.
tokenizer_cache = File.join(__dir__, "tokenizer_#{options[:vocab_size]}.json")

# ── Data ───────────────────────────────────────────────────────────────────────

files = options[:corpus].flat_map { |pat| Dir.glob(pat).sort }.uniq
raise "No corpus files found for: #{options[:corpus].join(", ")}" if files.empty?

puts "Loading corpus..."
corpus = files.each_with_object(+"") do |path, buf|
  text = File.read(path)
  puts "  #{File.basename(path)}: #{text.size} chars"
  buf << text
end
puts "  total: #{corpus.size} characters"
puts "  Pre-training on raw text. The data distribution is the prior — quality and diversity here determine the ceiling."

if !options[:retrain_tokenizer] && File.exist?(tokenizer_cache)
  begin
    tokenizer = Rinzler::Tokenizer::BPE.from_file(tokenizer_cache)
    if tokenizer.merges.size == options[:vocab_size]
      puts "\nTokenizer loaded from cache (#{tokenizer_cache})"
      puts "  #{tokenizer.vocab_size} tokens. Skipping BPE training — merge rules are deterministic given the same corpus and vocab size."
    else
      puts "\nCached tokenizer has #{tokenizer.merges.size} merges, #{options[:vocab_size]} requested — retraining."
      tokenizer = nil
    end
  rescue => e
    puts "\nCached tokenizer unreadable (#{e.message}) — retraining from scratch."
    tokenizer = nil
  end
end

unless tokenizer
  puts "\nTraining BPE tokenizer — #{options[:vocab_size]} merges (Sennrich et al., 2015)."
  puts "  Each merge reduces expected sequence length. Attention is O(n²) in sequence length — this is not cosmetic."
  puts "  Character-stream BPE (no word boundaries): spaces are first-class citizens in the vocabulary."
  tokenizer = Rinzler::Tokenizer::BPE.new.train(corpus, num_merges: options[:vocab_size])
  tokenizer.save(tokenizer_cache)
  puts "  Vocabulary: #{tokenizer.vocab_size} tokens (#{options[:vocab_size]} merges + #{tokenizer.vocab_size - options[:vocab_size]} base chars). Saved to #{tokenizer_cache}."
end

all_ids = tokenizer.encode(corpus)
puts "  #{all_ids.size} tokens total. Train/val split: 90/10."

# Split 90/10 train/val
split      = (all_ids.size * 0.9).to_i
train_ids  = all_ids[0...split]
val_ids    = all_ids[split..]

# Sample a batch of random windows from a token sequence.
# Returns a 2D array [batch_size][context_len + 1] — the extra token
# at the end is the final target.
def random_batch(ids, context_len, batch_size)
  batch_size.times.map do
    start = rand(ids.size - context_len - 1)
    ids[start...(start + context_len + 1)]
  end
end

# ── GPU backend (optional) ─────────────────────────────────────────────────────

if options[:vulkan]
  require "rinzler/tensor"
  require "rinzler/vulkan"
  Rinzler::Tensor.backend = :vulkan
  puts "\nGPU backend: Vulkan compute (#{Rinzler::Vulkan::SHADER_PATH})"
  puts "  Tiled GEMM shader handles forward-pass matmuls. Backward pass remains on CPU — gradients are cheap at this scale."
end

# ── Model ──────────────────────────────────────────────────────────────────────

puts "\nBuilding model..."

if options[:resume]
  puts "  Resuming from #{options[:resume]}..."
  model, start_step, opt_state = Rinzler::GPT::GPTModel.from_checkpoint(options[:resume])
  cfg = model.config
  puts "  Restored at step #{start_step}. Second-order optimizer moments are intact — loss of those would set the schedule back."
else
  start_step = 0
  cfg = Rinzler::GPT::Config.new(
    vocab_size:  tokenizer.vocab_size,
    context_len: options[:context],
    d_model:     options[:d_model],
    n_heads:     options[:n_heads],
    n_layers:    options[:n_layers],
    ffn_mult:    4
  )
  model = Rinzler::GPT::GPTModel.new(cfg)
  puts "  Fresh init. #{options[:n_layers]} decoder blocks, #{options[:n_heads]} heads, d_model=#{options[:d_model]}, ffn_mult=4."
  puts "  Weights initialised with scaled normal — early gradient signal depends on this being right."
end

n_params = model.parameters.sum { |p| p.data.size }
param_str = n_params.then { |n| n > 1_000_000 ? "#{(n/1e6).round(2)}M" : "#{n > 1000 ? "#{(n/1000.0).round(1)}k" : n}" }
puts "  #{param_str} parameters. Context: #{options[:context]} tokens. Batch: #{options[:batch_size]}."
puts "  Optimizer: AdamW, lr=#{options[:lr]}, weight_decay=0.1 (Loshchilov & Hutter, 2017). Decoupled decay avoids the Adam L2 conflation."

opt = Rinzler::Optim::AdamW.new(model.parameters, lr: options[:lr], weight_decay: 0.1)
opt.load_checkpoint_state!(opt_state) if options[:resume] && opt_state

if options[:warmup_steps] > 0
  scheduler = Rinzler::Optim::LinearWarmup.new(opt, warmup_steps: options[:warmup_steps])
  puts "  LR schedule: linear warmup over #{options[:warmup_steps]} steps → #{options[:lr]}."
else
  scheduler = opt
end

puts "  Gradient clipping: max_norm=#{options[:clip_grad]}." if options[:clip_grad]

# ── Training loop ──────────────────────────────────────────────────────────────

puts "\nTraining — #{options[:steps]} steps, cross-entropy loss, next-token prediction.\n\n"
start_time = Time.now

# Graceful shutdown on SIGINT (Ctrl-C) or SIGTERM.
# Sets a flag rather than raising — the loop checks it at the end of each step
# and breaks cleanly, then falls through to the final checkpoint save.
stop_requested = false
%w[INT TERM].each do |sig|
  Signal.trap(sig) do
    # Signal handlers run in a restricted context — no puts, no I/O.
    # Set flag only; the main loop does the rest.
    stop_requested = true
  end
end

# Divergence history: array of { step:, train:, val:, gap: } recorded at each eval.
# Used to compute the trend (Δgap) over the last --div-window eval windows.
div_history = []
last_step   = start_step

(start_step...options[:steps]).each do |step|
  batch = random_batch(train_ids, cfg.context_len, options[:batch_size])

  scheduler.zero_grad
  loss = model.loss(batch)
  loss.backward
  opt.clip_grad_norm!(options[:clip_grad]) if options[:clip_grad]
  scheduler.step

  loss_val = loss.data.sum.round(4)

  # ── Eval ─────────────────────────────────────────────────────────────────────
  if (step + 1) % options[:eval_every] == 0
    val_batch = random_batch(val_ids, cfg.context_len, options[:batch_size])
    val_loss  = model.loss(val_batch).data.sum.round(4)
    elapsed   = (Time.now - start_time).round(1)

    gap     = (val_loss - loss_val).round(4)
    gap_pct = loss_val > 0 ? (gap / loss_val * 100.0).round(1) : 0.0

    # Trend: change in gap vs div_window evals ago
    trend_str = ""
    if div_history.size >= options[:div_window]
      prev_gap  = div_history[-options[:div_window]][:gap]
      delta     = (gap - prev_gap).round(4)
      arrow     = delta > 0 ? "↑" : (delta < 0 ? "↓" : "→")
      trend_str = " | Δgap/#{options[:div_window]}: #{delta > 0 ? "+" : ""}#{delta}#{arrow}"
    end

    warn_str = gap_pct > options[:div_warn] ? " ⚠  generalization gap #{gap_pct}% — model is beginning to memorise training distribution" : ""

    puts "step #{step + 1}/#{options[:steps]} | train: #{loss_val} | val: #{val_loss} | gap: #{gap > 0 ? "+" : ""}#{gap} (#{gap_pct}%)#{trend_str} | #{elapsed}s#{warn_str}"

    div_history << { step: step + 1, train: loss_val, val: val_loss, gap: gap }

    if options[:div_crit] && gap_pct > options[:div_crit]
      puts "\nStopping early — divergence #{gap_pct}% exceeds critical threshold #{options[:div_crit]}% at step #{step + 1}."
      puts "The train/val gap has widened past the point where continued training is likely to improve generalisation."
      stop_requested = true
    end
  end

  # ── Generate sample ──────────────────────────────────────────────────────────
  if (step + 1) % options[:gen_every] == 0
    prompt = tokenizer.encode("The ")
    ids    = model.generate(prompt, max_new_tokens: options[:gen_len], temperature: 0.8)
    sample = tokenizer.decode(ids)
    puts "\n--- sample at step #{step + 1} (temperature=0.8, autoregressive) ---"
    puts sample
    puts "---\n\n"
  end

  # ── Checkpoint ───────────────────────────────────────────────────────────────
  if (step + 1) % options[:save_every] == 0
    ckpt = options[:out].sub(".json", "_step#{step + 1}.json")
    model.save_checkpoint(ckpt, step: step + 1, optimizer: opt)
    tokenizer.save(ckpt.sub(".json", "_tokenizer.json"))
    puts "Checkpoint: #{ckpt}"
  end

  last_step = step + 1

  if stop_requested
    puts "\nSignal received — stopping after step #{last_step}."
    break
  end
end

elapsed = (Time.now - start_time).round(1)
puts "\nTraining complete in #{elapsed}s."

model.save_checkpoint(options[:out], step: last_step, optimizer: opt)
tokenizer.save(options[:out].sub(".json", "_tokenizer.json"))
puts "Final checkpoint: #{options[:out]}"

# ── Generate final sample ──────────────────────────────────────────────────────
unless stop_requested
  puts "\n--- final generation (temperature=0.8) ---"
  prompt = tokenizer.encode("Ruby is ")
  ids    = model.generate(prompt, max_new_tokens: 300, temperature: 0.8)
  puts tokenizer.decode(ids)
  puts "---"
end
