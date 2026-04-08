# frozen_string_literal: true

require "json"

# Load the native extension if it has been compiled; fall back to pure Ruby.
begin
  require "rinzler/tokenizer/bpe_ext"
  BPE_NATIVE = true
rescue LoadError
  BPE_NATIVE = false
end

module Rinzler
  module Tokenizer
    # BPE — Byte Pair Encoding (Sennrich et al., 2015 for NLP; Gage 1994 originally)
    #
    # The compression algorithm that became the backbone of modern LLM tokenization.
    # Originally invented for data compression in 1994, repurposed for NLP in 2015,
    # and now used by GPT-2, GPT-3, GPT-4, LLaMA, and almost everything else.
    #
    # Why BPE over characters?
    #   The transformer's attention mechanism is O(n²) in sequence length.
    #   A 1000-character document is 1000 tokens at the character level.
    #   With BPE it might be ~250 tokens. That's a 16x reduction in attention cost.
    #   At GPT-3 scale, this difference is the difference between possible and impossible.
    #
    # How it works — the training algorithm:
    #   1. Start: every character is its own token. Corpus = list of token sequences.
    #   2. Count all adjacent pairs across the corpus.
    #   3. Merge the most frequent pair into a new token. Add it to vocabulary.
    #   4. Replace every occurrence of that pair in the corpus with the new token.
    #   5. Repeat until vocabulary reaches target size.
    #
    # Character-stream approach (vs word-level BPE):
    #   We do NOT split on whitespace before merging. Spaces are characters like any
    #   other — they get absorbed into tokens naturally (" the", "def ", "end\n").
    #   This is closer to how GPT-2's byte-level BPE works and is correct for code,
    #   where indentation and spacing are semantically meaningful.
    #
    #   Consequence: decode is just concatenation — tokens ARE the text, spaces included.
    #   No joining-with-spaces needed (or wanted).
    #
    # GPT-2's vocabulary: 50,257 tokens (50k BPE merges + 256 byte tokens + <endoftext>)
    class BPE
      SPECIAL_TOKENS = { "<PAD>" => 0, "<UNK>" => 1, "<BOS>" => 2, "<EOS>" => 3 }.freeze
      PAD_ID = 0
      UNK_ID = 1
      BOS_ID = 2
      EOS_ID = 3

      attr_reader :vocab_size, :stoi, :itos, :merges

      def initialize
        @stoi       = {}
        @itos       = {}
        @merges     = {}   # {[token_a, token_b] => merged_token}
        @vocab_size = 0
      end

      # Train BPE on a corpus string, learning `num_merges` merge rules.
      # More merges = larger vocabulary = longer tokens = shorter sequences.
      #
      # We split the corpus into lines (rather than one huge sequence) for
      # efficiency — pair counting stays fast without sacrificing quality,
      # since most meaningful subwords occur within a line.
      def train(text, num_merges: 500)
        SPECIAL_TOKENS.each { |token, id| register(token, id) }

        # Seed: every unique character gets its own token
        text.chars.uniq.sort.each do |c|
          register(c, @stoi.size) unless @stoi.key?(c)
        end

        # Working corpus: one token-ID sequence per line
        corpus = text.lines.map { |line| line.chars.map { |c| @stoi[c] } }
                           .reject(&:empty?)

        if BPE_NATIVE
          train_fast_native(corpus, num_merges)
        else
          train_slow(corpus, num_merges)
        end

        @vocab_size = @stoi.size
        self
      end

      # Encode a string by applying learned merges greedily.
      # The input is treated as a flat character stream — no word splitting.
      def encode(text, add_bos: false, add_eos: false)
        ids = text.chars.map { |c| @stoi.fetch(c, UNK_ID) }

        @merges.each do |pair, merged_id|
          ids = merge_pair(ids, pair, merged_id)
        end

        ids.prepend(BOS_ID) if add_bos
        ids.append(EOS_ID)  if add_eos
        ids
      end

      # Decode token IDs back to a string.
      # Tokens already contain spaces (they're part of the learned subwords),
      # so we concatenate directly — no joining with spaces.
      def decode(ids, keep_special: false)
        tokens = ids.map { |id| @itos.fetch(id, "") }
        tokens = tokens.reject { |t| SPECIAL_TOKENS.key?(t) } unless keep_special
        tokens.join
      end

      def to_json(*) = JSON.generate({
        stoi:   @stoi,
        itos:   @itos.transform_keys(&:to_s),
        merges: @merges.map { |pair, id| [pair.join(","), id] }.to_h
      })

      def self.from_json(json)
        data = JSON.parse(json)
        t    = new
        t.instance_variable_set(:@stoi, data["stoi"])
        t.instance_variable_set(:@itos, data["itos"].transform_keys(&:to_i))
        t.instance_variable_set(:@merges,
          (data["merges"] || {}).transform_keys { |k| k.split(",").map(&:to_i) }
                                .transform_values(&:to_i))
        t.instance_variable_set(:@vocab_size, data["stoi"].size)
        t
      end

      def self.from_file(path) = from_json(File.read(path))
      def save(path)           = File.write(path, to_json)

      private

      # Fast path: one C call to train_fast returns the merge list in order.
      # We register each merged token in Ruby (string concatenation + vocab bookkeeping).
      def train_fast_native(corpus, num_merges)
        vocab_base = @stoi.size
        merge_pairs = Rinzler::Tokenizer::BPEExt.train_fast(corpus, num_merges, vocab_base)
        $stdout.puts  # newline after C's progress line

        merge_pairs.each_with_index do |(a, b), i|
          new_token = @itos[a] + @itos[b]
          new_id    = vocab_base + i
          register(new_token, new_id)
          @merges[[a, b]] = new_id
        end
      end

      # Slow path: pure Ruby, naive O(N × merges). Kept as reference + fallback.
      def train_slow(corpus, num_merges)
        report_every = [num_merges / 10, 1].max
        num_merges.times do |i|
          pairs = count_pairs(corpus)
          break if pairs.empty?

          best_pair = pairs.max_by { |_, count| count }.first
          new_token = @itos[best_pair[0]] + @itos[best_pair[1]]
          new_id    = @stoi.size
          register(new_token, new_id)
          @merges[best_pair] = new_id
          corpus = corpus.map { |seq| merge_pair(seq, best_pair, new_id) }

          if (i + 1) % report_every == 0 || i + 1 == num_merges
            pct = ((i + 1) * 100.0 / num_merges).round
            $stdout.print "\r  BPE: #{i + 1}/#{num_merges} merges (#{pct}%)  vocab=#{@stoi.size}"
            $stdout.flush
          end
        end
        $stdout.puts
      end

      def register(token, id)
        @stoi[token] = id
        @itos[id]    = token
      end

      # Count all adjacent pairs across all sequences in the corpus.
      #
      # Native path: ~10-20x faster than pure Ruby for large corpora — avoids
      # per-pair Array allocation in the hot loop and keeps everything off the GIL.
      # Pure Ruby fallback is the reference implementation; behaviour is identical.
      if BPE_NATIVE
        def count_pairs(corpus) = Rinzler::Tokenizer::BPEExt.count_pairs(corpus)
      else
        def count_pairs(corpus)
          counts = Hash.new(0)
          corpus.each do |seq|
            seq.each_cons(2) { |pair| counts[pair] += 1 }
          end
          counts
        end
      end

      # Replace every occurrence of `pair` in `seq` with `new_id`.
      if BPE_NATIVE
        def merge_pair(seq, pair, new_id) = Rinzler::Tokenizer::BPEExt.merge_pair(seq, pair[0], pair[1], new_id)
      else
        def merge_pair(seq, pair, new_id)
          result = []
          i = 0
          while i < seq.size
            if i < seq.size - 1 && seq[i] == pair[0] && seq[i + 1] == pair[1]
              result << new_id
              i += 2
            else
              result << seq[i]
              i += 1
            end
          end
          result
        end
      end
    end
  end
end
