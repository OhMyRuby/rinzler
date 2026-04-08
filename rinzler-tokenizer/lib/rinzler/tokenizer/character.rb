# frozen_string_literal: true

require "json"

module Rinzler
  module Tokenizer
    # CharacterTokenizer — the simplest possible tokenizer.
    #
    # Every unique character in the training corpus gets an integer ID.
    # "hello" → [h=15, e=8, l=19, l=19, o=22]
    #
    # This is where tokenization began. Early neural language models (Bengio 2003,
    # Sutskever's char-RNN 2011) operated at the character level because it
    # required no preprocessing and handled any text naturally.
    #
    # The tradeoff: sequences are long. "the quick brown fox" is 19 tokens at
    # the character level, maybe 4 at the word level, maybe 6 with BPE.
    # Transformers scale quadratically with sequence length (attention is O(n²)),
    # so long sequences are expensive. That's the pressure that led to BPE.
    #
    # But character-level has real advantages too:
    #   - Perfect coverage: no unknown tokens, ever
    #   - Tiny vocabulary (typically 65-200 tokens vs 50k+ for BPE)
    #   - The model must learn spelling, morphology, everything from scratch —
    #     which means it actually learns them, deeply
    #
    # For learning purposes, character-level is ideal. You can see exactly
    # what the model is predicting at every step.
    class Character
      # Special tokens that exist outside the normal vocabulary.
      # PAD fills sequences to equal length in a batch.
      # UNK represents characters seen at inference but not in training.
      # BOS/EOS mark the boundaries of a sequence.
      SPECIAL_TOKENS = { "<PAD>" => 0, "<UNK>" => 1, "<BOS>" => 2, "<EOS>" => 3 }.freeze
      PAD_ID = 0
      UNK_ID = 1
      BOS_ID = 2
      EOS_ID = 3

      attr_reader :vocab_size, :stoi, :itos

      def initialize
        @stoi = {}   # string → integer
        @itos = {}   # integer → string
        @vocab_size = 0
      end

      # Build the vocabulary from a corpus string.
      # Call this once on your training data before encoding anything.
      def train(text)
        SPECIAL_TOKENS.each { |token, id| register(token, id) }

        text.chars.uniq.sort.each do |char|
          register(char, @stoi.size) unless @stoi.key?(char)
        end

        @vocab_size = @stoi.size
        self
      end

      # Encode a string to an array of integer token IDs.
      # Unknown characters map to UNK_ID.
      def encode(text, add_bos: false, add_eos: false)
        ids = text.chars.map { |c| @stoi.fetch(c, UNK_ID) }
        ids.prepend(BOS_ID) if add_bos
        ids.append(EOS_ID)  if add_eos
        ids
      end

      # Decode an array of integer token IDs back to a string.
      # Special tokens are dropped unless keep_special: true.
      def decode(ids, keep_special: false)
        tokens = ids.map { |id| @itos.fetch(id, "<UNK>") }
        tokens = tokens.reject { |t| SPECIAL_TOKENS.key?(t) } unless keep_special
        tokens.join
      end

      # Serialize vocabulary to JSON for saving alongside a trained model.
      def to_json(*) = JSON.generate({ stoi: @stoi, itos: @itos.transform_keys(&:to_s) })

      # Restore a tokenizer from saved JSON.
      def self.from_json(json)
        data    = JSON.parse(json)
        t       = new
        t.instance_variable_set(:@stoi, data["stoi"])
        t.instance_variable_set(:@itos, data["itos"].transform_keys(&:to_i))
        t.instance_variable_set(:@vocab_size, data["stoi"].size)
        t
      end

      def self.from_file(path) = from_json(File.read(path))

      def save(path) = File.write(path, to_json)

      private

      def register(token, id)
        @stoi[token] = id
        @itos[id]    = token
      end
    end
  end
end
