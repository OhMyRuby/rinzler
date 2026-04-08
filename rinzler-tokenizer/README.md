# rinzler-tokenizer

Byte Pair Encoding tokenizer. Converts raw text into integer token IDs for the GPT model.

## What it does

BPE (Sennrich et al., 2015) starts with every character as its own token, then iteratively merges the most frequent adjacent pair into a new token. After `n` merges, sequences that were `n` characters long can be a single token. Because attention is O(sequence_length²), shorter sequences are dramatically cheaper.

**Character-stream approach:** spaces are not split on before merging. Tokens naturally absorb surrounding whitespace (`" the"`, `"def "`, `"end\n"`). Decode is just concatenation — no joining logic needed.

## Usage

```ruby
require "rinzler/tokenizer"

# Train
tokenizer = Rinzler::Tokenizer::BPE.new.train(corpus_text, num_merges: 1000)
tokenizer.save("tokenizer.json")

# Load from cache
tokenizer = Rinzler::Tokenizer::BPE.from_file("tokenizer.json")

# Encode / decode
ids  = tokenizer.encode("def hello_world")   # → [42, 17, 305, ...]
text = tokenizer.decode(ids)                 # → "def hello_world"

tokenizer.vocab_size   # num_merges + unique base characters + 4 special tokens
```

## Special tokens

| Token | ID | Use |
|-------|----|-----|
| `<PAD>` | 0 | Padding |
| `<UNK>` | 1 | Unknown |
| `<BOS>` | 2 | Beginning of sequence |
| `<EOS>` | 3 | End of sequence |

## Native extension

The merge loop has a C extension (`bpe_ext`) for performance. If it hasn't been compiled, the tokenizer falls back to pure Ruby automatically. Build with `bundle exec rake compile` inside `rinzler-tokenizer/`.
