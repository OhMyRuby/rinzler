# frozen_string_literal: true

require "json"

module Rinzler
  module GPT
    # Config holds the hyperparameters that define a GPT model's size and behaviour.
    #
    # These numbers directly control the capacity (and cost) of the model:
    #   - vocab_size:    how many tokens the model knows (from the tokenizer)
    #   - context_len:   maximum sequence length (attention is O(n²), so this matters)
    #   - d_model:       embedding dimension — the width of every layer
    #   - n_heads:       attention heads per layer (d_model must be divisible by this)
    #   - n_layers:      how many transformer blocks to stack (depth)
    #   - ffn_mult:      FFN inner dimension multiplier (typically 4)
    #
    # Rough scale guide (for character-level models):
    #   Tiny   (laptop, minutes):  d=64,  heads=4, layers=4,  ctx=128
    #   Small  (laptop, hours):    d=128, heads=4, layers=6,  ctx=256
    #   Medium (GPU, hours):       d=256, heads=8, layers=8,  ctx=512
    Config = Data.define(:vocab_size, :context_len, :d_model, :n_heads, :n_layers, :ffn_mult) do
      def self.tiny(vocab_size)
        new(vocab_size:, context_len: 128, d_model: 64, n_heads: 4, n_layers: 4, ffn_mult: 4)
      end

      def self.small(vocab_size)
        new(vocab_size:, context_len: 256, d_model: 128, n_heads: 4, n_layers: 6, ffn_mult: 4)
      end
    end

    # GPTModel — the full language model.
    #
    # Architecture (GPT-2 style):
    #
    #   token_ids  →  Token Embedding  ─┐
    #   positions  →  Position Embedding ┘  (sum)
    #                    ↓
    #              [TransformerBlock] × n_layers
    #                    ↓
    #               LayerNorm
    #                    ↓
    #              Linear → logits [seq_len, vocab_size]
    #
    # Token embeddings: each token ID maps to a learned vector. The model learns
    # what each token "means" in the embedding space.
    #
    # Position embeddings: transformers have no inherent sense of order — attention
    # treats the sequence as a set. We add learned position vectors so the model
    # knows where each token sits. Position 0 gets one vector, position 1 another, etc.
    # (GPT-2 uses learned positions; RoPE and ALiBi are more modern alternatives
    # that generalize better to lengths beyond training — future work for Rinzler.)
    #
    # The output logits are raw scores over the vocabulary. During training we apply
    # cross-entropy loss. During inference we apply softmax and sample.
    #
    # Batching: forward and loss accept either a 1D Array (single sequence) or
    # a 2D Array of Arrays (batch). generate always takes a single 1D sequence.
    class GPTModel < NN::Module
      attr_reader :config

      def initialize(config)
        @config = config

        @token_emb = NN::Embedding.new(config.vocab_size, config.d_model)
        @pos_emb   = NN::Embedding.new(config.context_len, config.d_model)

        @blocks    = Array.new(config.n_layers) do
          TransformerBlock.new(config.d_model, config.n_heads, ffn_mult: config.ffn_mult, context_len: config.context_len)
        end

        @ln_final  = NN::LayerNorm.new(config.d_model)
        @lm_head   = NN::Linear.new(config.d_model, config.vocab_size, bias: false)
      end

      # Forward pass.
      #
      # token_ids: 1D Array [T] or 2D Array [B][T] of Integer token IDs
      # Returns:   Tensor [T, vocab_size] for 1D input
      #            Tensor [B, T, vocab_size] for 2D input
      def forward(token_ids)
        batched = token_ids.first.is_a?(Array)
        ids     = batched ? token_ids : [token_ids]
        b, t    = ids.size, ids[0].size

        raise ArgumentError, "sequence length #{t} exceeds context_len #{@config.context_len}" \
          if t > @config.context_len

        # Embed tokens: [B, T, d_model]
        tok = @token_emb.call(ids)

        # Position embeddings: [T, d_model] — broadcasts to [B, T, d_model] on addition.
        # Each sequence in the batch sees the same position vectors, which is correct:
        # position 5 means the same thing regardless of which batch item we're in.
        pos = @pos_emb.call((0...t).to_a)
        x   = tok + pos                         # [B, T, d_model]

        # Run through all transformer blocks
        @blocks.each { |block| x = block.call(x) }

        # Final norm and project to vocabulary: [B, T, vocab_size]
        logits = @lm_head.call(@ln_final.call(x))

        # Squeeze batch dim for single-sequence input
        batched ? logits : logits.reshape(t, @config.vocab_size)
      end

      # Compute cross-entropy loss for next-token prediction.
      #
      # Vectorized: instead of looping over B×T positions and building a separate
      # Tensor for each scalar NLL, we compute log-softmax for every position at
      # once, then use a one-hot mask to select the correct token's log-probability.
      #
      #   loss = -mean( log_softmax(logits)[b, t, target[b,t]]  for all b, t )
      #
      # The one-hot matrix is a constant (no gradient flows through it) — it's just
      # a selector. The gradient path is:
      #
      #   mean → sum(axis:1) → log_softmax → reshape → forward
      #
      # Collapsed from B×T Tensor allocations to ~5 matrix ops.
      #
      # token_ids: 1D Array [T+1] or 2D Array [B][T+1]
      def loss(token_ids)
        batched = token_ids.first.is_a?(Array)
        seqs    = batched ? token_ids : [token_ids]

        inputs  = seqs.map { |s| s[0...-1] }   # [B][T] — context
        targets = seqs.map { |s| s[1..] }       # [B][T] — next tokens

        b, t   = seqs.size, targets[0].size
        logits = forward(inputs)                # [B, T, vocab_size]

        # Flatten to [B*T, vocab_size] for row-wise log-softmax
        log_probs = logits.reshape(b * t, @config.vocab_size).log_softmax

        # One-hot target mask [B*T, vocab_size] — constant, no gradient needed.
        # Row i has a single 1 at the column of the correct next token.
        targets_flat = targets.flatten
        one_hot      = Numo::DFloat.zeros(b * t, @config.vocab_size)
        targets_flat.each_with_index { |tid, i| one_hot[i, tid] = 1.0 }

        # Gather: multiply log_probs by the one-hot mask and sum across vocab dim.
        # Each row collapses to a single scalar: the log-prob of the correct token.
        nll = (log_probs * Rinzler::Tensor::Tensor.new(one_hot)).sum(axis: 1)

        -nll.mean
      end

      # Persist the model to disk.
      #
      # Writes two files:
      #   <path>.bin  — binary payload: 4-byte big-endian uint32 version header,
      #                 then raw parameter bytes (Numo DFloat native encoding),
      #                 then optimizer m/v buffer bytes if present.
      #   <path>.json — JSON sidecar: config, step, parameter shapes (in order),
      #                 optimizer shapes and step_count if present.
      #
      # The version header allows format evolution while preserving backwards
      # compatibility. Current format version: 1.
      CHECKPOINT_VERSION = 1

      def save_checkpoint(path, step:, optimizer: nil)
        bin_path  = path.sub(/\.json$/, "") + ".bin"
        json_path = path.sub(/\.json$/, "") + ".json"

        params     = parameters
        opt_state  = optimizer&.checkpoint_state

        # ── Binary payload ────────────────────────────────────────────────────
        File.open(bin_path, "wb") do |f|
          f.write([CHECKPOINT_VERSION].pack("N"))   # 4-byte big-endian version

          params.each { |p| f.write(p.data.to_string) }

          if opt_state
            opt_state["m"].each { |buf| f.write(buf.to_string) }
            opt_state["v"].each { |buf| f.write(buf.to_string) }
          end
        end

        # ── JSON sidecar ──────────────────────────────────────────────────────
        meta = {
          "version"    => CHECKPOINT_VERSION,
          "step"       => step,
          "config"     => {
            "vocab_size"  => @config.vocab_size,
            "context_len" => @config.context_len,
            "d_model"     => @config.d_model,
            "n_heads"     => @config.n_heads,
            "n_layers"    => @config.n_layers,
            "ffn_mult"    => @config.ffn_mult
          },
          "parameters" => params.map { |p| { "shape" => p.shape } }
        }

        if opt_state
          meta["optimizer"] = {
            "step_count" => opt_state["step_count"],
            "m"          => opt_state["m"].map { |buf| { "shape" => buf.shape } },
            "v"          => opt_state["v"].map { |buf| { "shape" => buf.shape } }
          }
        end

        File.write(json_path, JSON.generate(meta))
      end

      # Restore a model (and optionally optimizer state) from a checkpoint.
      # Accepts either the .json or .bin path — both are resolved from the stem.
      # Returns [model, step, optimizer_state_hash_or_nil].
      def self.from_checkpoint(path)
        stem      = path.sub(/\.(json|bin)$/, "")
        bin_path  = stem + ".bin"
        json_path = stem + ".json"

        # ── Legacy JSON-only format ───────────────────────────────────────────
        unless File.exist?(bin_path)
          raw   = JSON.parse(File.read(json_path))
          cfg   = Config.new(**raw["config"].transform_keys(&:to_sym))
          model = new(cfg)
          model.parameters.each_with_index do |p, i|
            saved  = raw["parameters"][i]
            p.data = Numo::DFloat.cast(saved["data"]).reshape(*saved["shape"])
          end
          Rinzler.logger.info("Checkpoint loaded (legacy JSON) — step #{raw["step"]}, path: #{json_path}")
          return [model, raw["step"], raw["optimizer"]]
        end

        # ── Binary format ─────────────────────────────────────────────────────
        meta = JSON.parse(File.read(json_path))
        _version = meta["version"]   # reserved for future format migrations

        cfg   = Config.new(**meta["config"].transform_keys(&:to_sym))
        model = new(cfg)

        opt_state = nil
        File.open(bin_path, "rb") do |f|
          f.read(4).unpack1("N")   # consume version header

          model.parameters.each_with_index do |p, i|
            shape     = meta["parameters"][i]["shape"]
            byte_size = shape.reduce(1, :*) * Numo::DFloat::ELEMENT_BYTE_SIZE
            p.data    = Numo::DFloat.from_string(f.read(byte_size)).reshape(*shape)
          end

          if meta["optimizer"]
            opt_meta = meta["optimizer"]
            m_arrays = opt_meta["m"].map do |m|
              shape     = m["shape"]
              byte_size = shape.reduce(1, :*) * Numo::DFloat::ELEMENT_BYTE_SIZE
              Numo::DFloat.from_string(f.read(byte_size)).reshape(*shape)
            end
            v_arrays = opt_meta["v"].map do |v|
              shape     = v["shape"]
              byte_size = shape.reduce(1, :*) * Numo::DFloat::ELEMENT_BYTE_SIZE
              Numo::DFloat.from_string(f.read(byte_size)).reshape(*shape)
            end
            opt_state = { "step_count" => opt_meta["step_count"], "m" => m_arrays, "v" => v_arrays }
          end
        end

        Rinzler.logger.info("Checkpoint loaded — step #{meta["step"]}, path: #{bin_path}")
        [model, meta["step"], opt_state]
      end

      # Generate `max_new_tokens` tokens autoregressively given a prompt.
      # temperature > 1.0 = more random; < 1.0 = more conservative.
      def generate(token_ids, max_new_tokens:, temperature: 1.0)
        ids = token_ids.dup

        max_new_tokens.times do
          # Crop context to window
          ctx      = ids.last(@config.context_len)
          logits   = forward(ctx)               # 1D input → [T, vocab_size]

          # Take the last position's logits (next-token prediction)
          last_row = logits.data[-1, true]

          # Sample from the distribution
          ids << sample(last_row, temperature:)
        end

        ids
      end

      private

      def sample(logits, temperature:)
        scaled = logits / temperature
        max    = scaled.max
        probs  = Numo::NMath.exp(scaled - max)
        probs  = probs / probs.sum

        # Multinomial sample
        r   = rand
        cum = 0.0
        probs.each_with_index do |p, i|
          cum += p
          return i if cum >= r
        end
        probs.size - 1
      end
    end
  end
end
