# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/rinzler/tokenizer"

include Rinzler::Tokenizer

CORPUS = "the quick brown fox jumps over the lazy dog"

class TestCharacterTokenizer < Minitest::Test
  def setup
    @t = Character.new.train(CORPUS)
  end

  def test_vocab_includes_all_chars
    CORPUS.chars.uniq.each do |c|
      assert @t.stoi.key?(c), "vocab missing char: #{c.inspect}"
    end
  end

  def test_vocab_includes_special_tokens
    assert @t.stoi.key?("<PAD>")
    assert @t.stoi.key?("<UNK>")
    assert @t.stoi.key?("<BOS>")
    assert @t.stoi.key?("<EOS>")
  end

  def test_encode_decode_roundtrip
    text    = "the fox"
    encoded = @t.encode(text)
    decoded = @t.decode(encoded)
    assert_equal text, decoded
  end

  def test_encode_length_matches_chars
    assert_equal 7, @t.encode("the fox").size
  end

  def test_add_bos_eos
    ids = @t.encode("hi", add_bos: true, add_eos: true)
    assert_equal Character::BOS_ID, ids.first
    assert_equal Character::EOS_ID, ids.last
    assert_equal 4, ids.size  # BOS + h + i + EOS
  end

  def test_unknown_char_maps_to_unk
    ids = @t.encode("€")
    assert_equal Character::UNK_ID, ids.first
  end

  def test_decode_drops_special_tokens_by_default
    ids    = @t.encode("hi", add_bos: true, add_eos: true)
    result = @t.decode(ids)
    assert_equal "hi", result
  end

  def test_decode_keeps_special_tokens_when_asked
    ids    = @t.encode("hi", add_bos: true, add_eos: true)
    result = @t.decode(ids, keep_special: true)
    assert result.include?("<BOS>")
    assert result.include?("<EOS>")
  end

  def test_json_roundtrip
    json = @t.to_json
    t2   = Character.from_json(json)
    assert_equal @t.encode("the fox"), t2.encode("the fox")
  end

  def test_vocab_size
    # 4 special tokens + unique chars in corpus
    expected = 4 + CORPUS.chars.uniq.size
    assert_equal expected, @t.vocab_size
  end
end

class TestBPETokenizer < Minitest::Test
  def setup
    # Use a repetitive corpus so BPE finds clear merge candidates
    text = "the the the cat cat sat sat on the mat mat mat " * 20
    @t   = BPE.new.train(text, num_merges: 20)
  end

  def test_vocab_grows_with_merges
    # Vocabulary should be larger than just characters + specials
    char_count = "the cat sat on the mat".chars.uniq.size + 4
    assert @t.vocab_size > char_count
  end

  def test_frequent_pairs_get_merged
    # "th" should be a learned token since "the" repeats heavily
    assert @t.stoi.key?("th") || @t.stoi.key?("the"),
           "frequent subword should be in vocabulary"
  end

  def test_encode_returns_integers
    ids = @t.encode("the cat")
    assert ids.all? { |id| id.is_a?(Integer) }
  end

  def test_encode_fewer_tokens_than_chars
    text      = "the the the"
    char_ids  = text.chars.size
    bpe_ids   = @t.encode(text).size
    assert bpe_ids < char_ids,
           "BPE should produce fewer tokens than characters (#{bpe_ids} vs #{char_ids})"
  end

  def test_add_bos_eos
    ids = @t.encode("the", add_bos: true, add_eos: true)
    assert_equal BPE::BOS_ID, ids.first
    assert_equal BPE::EOS_ID, ids.last
  end

  def test_json_roundtrip
    text  = "the cat sat"
    json  = @t.to_json
    t2    = BPE.from_json(json)
    assert_equal @t.encode(text), t2.encode(text)
  end

  def test_merges_reduce_sequence_length
    # A highly repetitive string should compress well
    text     = "the " * 50
    char_len = text.gsub(" ", "").chars.size
    bpe_len  = @t.encode(text).size
    assert bpe_len < char_len,
           "BPE encoding should be shorter than character encoding"
  end
end
