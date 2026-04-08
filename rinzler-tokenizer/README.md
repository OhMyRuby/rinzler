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

## Native extension (bpe_ext)

The merge loop has a C extension (`bpe_ext`) for performance. If it hasn't been compiled, the tokenizer falls back to pure Ruby automatically. Build with `bundle exec rake compile` inside `rinzler-tokenizer/`.

### Data structures

The extension maintains a doubly-linked list of `Cell` nodes (each holding a token ID and prev/next pointers into a contiguous `Cell[]` arena) plus an inverted `OccurrenceIndex` — a hash map from pair keys to `CellVec` arrays of cell positions where that pair occurs. Finding the best pair each merge step is O(vocab) in the index; applying the merge is O(occurrences).

### Known bug fixed: use-after-free on idx_grow

When the occurrence index hash table grew (`idx_grow`), it freed and reallocated the `vecs` container array. Any `CellVec*` pointer obtained before the grow was dangling afterward. This caused a segfault consistently around 90% of the merge loop (when the index grew past capacity at ~900/1000 merges).

Fix: snapshot `occ->len` and `occ->data` before the inner merge loop. The `data` arrays (individual `CellVec` payloads) survive the grow — only the `vecs` container is replaced. The snapshot pointers remain valid for the duration of the inner loop.

```c
CellVec *occ = idx_get(&index, best_key);
if (occ) {
    int32_t  occ_len  = occ->len;
    int32_t *occ_data = occ->data;
    for (int32_t oi = 0; oi < occ_len; oi++) {
        int32_t ci_a = occ_data[oi];
        /* inner merge logic — may trigger idx_push → idx_grow */
    }
}
```

## Future optimization paths

- **Heap-based best-pair selection** — currently scans all pairs each step to find the maximum frequency (O(vocab)). A max-heap or priority queue with lazy deletion would give O(log vocab) per step, meaningful at vocab_size > 5000.
- **Parallel pair counting** — the initial character-frequency pass over a large corpus is single-threaded. A parallel reduce over corpus chunks (Ractors or C threads) could cut startup time for large corpora.
- **Incremental index updates** — after a merge, only pairs adjacent to the merged positions change frequency. The current implementation rebuilds affected entries correctly; a skip-list or finger-tree over the token stream could make these updates O(1) per affected site rather than O(occurrences).
