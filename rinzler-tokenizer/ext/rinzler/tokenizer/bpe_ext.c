/*
 * bpe_ext.c — Native acceleration for BPE tokenizer hot loops
 *
 * Exposes two methods on Rinzler::Tokenizer::BPE::Ext:
 *   count_pairs(corpus)              → Hash { [a,b] => count }
 *   merge_pair(seq, pair_a, pair_b, new_id) → Array
 *
 * These are the two inner loops that dominate BPE training time.
 * Each merge iteration calls count_pairs once (full corpus scan) and
 * merge_pair once per sequence in the corpus — pure Ruby for 1000 merges
 * over a 2M-character corpus is painfully slow. C is not.
 *
 * Design note: we keep the Ruby BPE class intact as the reference
 * implementation. This extension is loaded opportunistically; if it's
 * absent the Ruby fallback runs transparently. The Ruby methods are
 * the spec; these are the fast path.
 */

#include "ruby.h"
#include <stdint.h>

/* ── count_pairs ──────────────────────────────────────────────────────────────
 *
 * Walk every sequence in the corpus and tally adjacent (a, b) pairs.
 *
 * corpus: Array of Arrays of Integers (token IDs)
 * returns: Hash mapping [a, b] (2-element Ruby Array) → Integer count
 *
 * We use a flat hash keyed by a 64-bit packed pair (a<<32|b) internally,
 * then convert to Ruby Arrays at the end. This avoids allocating a Ruby
 * Array per pair per occurrence during the hot inner loop.
 */

/* Simple open-addressing hash table for uint64→int64 counts. */
#define HT_INIT_CAP 65536

typedef struct {
    uint64_t *keys;
    int64_t  *vals;
    uint32_t  cap;
    uint32_t  size;
} PairTable;

static void pt_init(PairTable *pt) {
    pt->cap  = HT_INIT_CAP;
    pt->size = 0;
    pt->keys = calloc(pt->cap, sizeof(uint64_t));
    pt->vals = calloc(pt->cap, sizeof(int64_t));
    if (!pt->keys || !pt->vals) rb_raise(rb_eNoMemError, "count_pairs: OOM");
    /* sentinel: 0 means empty. Token IDs are non-negative; pack stores (a+1)<<32|(b+1)
     * so 0 is never a valid key. */
}

static void pt_free(PairTable *pt) {
    free(pt->keys);
    free(pt->vals);
}

static void pt_grow(PairTable *pt) {
    uint32_t  old_cap  = pt->cap;
    uint64_t *old_keys = pt->keys;
    int64_t  *old_vals = pt->vals;

    pt->cap  = old_cap * 2;
    pt->keys = calloc(pt->cap, sizeof(uint64_t));
    pt->vals = calloc(pt->cap, sizeof(int64_t));
    if (!pt->keys || !pt->vals) rb_raise(rb_eNoMemError, "count_pairs: OOM on grow");

    for (uint32_t i = 0; i < old_cap; i++) {
        if (old_keys[i] == 0) continue;
        uint32_t slot = (uint32_t)(old_keys[i] ^ (old_keys[i] >> 17)) & (pt->cap - 1);
        while (pt->keys[slot]) slot = (slot + 1) & (pt->cap - 1);
        pt->keys[slot] = old_keys[i];
        pt->vals[slot] = old_vals[i];
    }

    free(old_keys);
    free(old_vals);
}

/* Increment count for packed key. key must be non-zero. */
static void pt_inc(PairTable *pt, uint64_t key) {
    if (pt->size * 2 >= pt->cap) pt_grow(pt);

    uint32_t slot = (uint32_t)(key ^ (key >> 17)) & (pt->cap - 1);
    while (pt->keys[slot] && pt->keys[slot] != key)
        slot = (slot + 1) & (pt->cap - 1);

    if (!pt->keys[slot]) {
        pt->keys[slot] = key;
        pt->size++;
    }
    pt->vals[slot]++;
}

static VALUE rb_count_pairs(VALUE self, VALUE corpus) {
    Check_Type(corpus, T_ARRAY);

    PairTable pt;
    pt_init(&pt);

    long n_seqs = RARRAY_LEN(corpus);
    for (long s = 0; s < n_seqs; s++) {
        VALUE seq = RARRAY_AREF(corpus, s);
        Check_Type(seq, T_ARRAY);
        long len = RARRAY_LEN(seq);
        if (len < 2) continue;

        for (long i = 0; i < len - 1; i++) {
            /* Unpack token IDs; offset by 1 so 0 is the sentinel. */
            long a = NUM2LONG(RARRAY_AREF(seq, i));
            long b = NUM2LONG(RARRAY_AREF(seq, i + 1));
            uint64_t key = ((uint64_t)(a + 1) << 32) | (uint64_t)(b + 1);
            pt_inc(&pt, key);
        }
    }

    /* Convert to Ruby Hash { [a, b] => count } */
    VALUE result = rb_hash_new();
    for (uint32_t i = 0; i < pt.cap; i++) {
        if (!pt.keys[i]) continue;
        long a = (long)((pt.keys[i] >> 32) & 0xFFFFFFFF) - 1;
        long b = (long)(pt.keys[i] & 0xFFFFFFFF) - 1;
        VALUE pair = rb_ary_new_from_args(2, LONG2NUM(a), LONG2NUM(b));
        rb_hash_aset(result, pair, LONG2NUM(pt.vals[i]));
    }

    pt_free(&pt);
    return result;
}

/* ── merge_pair ───────────────────────────────────────────────────────────────
 *
 * Replace every occurrence of the adjacent pair (pair_a, pair_b) in seq
 * with new_id. Single linear scan, no allocation until we know the output size.
 *
 * seq:    Array of Integers
 * pair_a: Integer
 * pair_b: Integer
 * new_id: Integer
 * returns: new Array of Integers
 */
static VALUE rb_merge_pair(VALUE self, VALUE seq, VALUE r_pair_a, VALUE r_pair_b, VALUE r_new_id) {
    Check_Type(seq, T_ARRAY);

    long len    = RARRAY_LEN(seq);
    long pair_a = NUM2LONG(r_pair_a);
    long pair_b = NUM2LONG(r_pair_b);
    long new_id = NUM2LONG(r_new_id);

    /* Upper bound: same length as input (no merges found). */
    VALUE result = rb_ary_new_capa(len);

    long i = 0;
    while (i < len) {
        if (i < len - 1 &&
            NUM2LONG(RARRAY_AREF(seq, i))     == pair_a &&
            NUM2LONG(RARRAY_AREF(seq, i + 1)) == pair_b) {
            rb_ary_push(result, LONG2NUM(new_id));
            i += 2;
        } else {
            rb_ary_push(result, RARRAY_AREF(seq, i));
            i += 1;
        }
    }

    return result;
}

/* ── Init ─────────────────────────────────────────────────────────────────── */

void Init_bpe_ext(void) {
    VALUE mRinzler   = rb_define_module("Rinzler");
    VALUE mTokenizer = rb_define_module_under(mRinzler, "Tokenizer");
    VALUE mExt       = rb_define_module_under(mTokenizer, "BPEExt");

    rb_define_module_function(mExt, "count_pairs", rb_count_pairs, 1);
    rb_define_module_function(mExt, "merge_pair",  rb_merge_pair,  4);
}
