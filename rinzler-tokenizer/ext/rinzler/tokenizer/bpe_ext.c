/*
 * bpe_ext.c — Native acceleration for BPE tokenizer hot loops
 *
 * Methods on Rinzler::Tokenizer::BPEExt:
 *
 *   count_pairs(corpus)                              → Hash {[a,b] => count}
 *   merge_pair(seq, pair_a, pair_b, new_id)          → Array
 *   train_fast(corpus, num_merges, vocab_base)       → Array of [a, b] pairs
 *
 * train_fast — fast O(N + merges·k̄) BPE using a pair index:
 *
 *   Naive BPE rescans the entire corpus (O(N)) on every merge to recount pairs
 *   and apply the substitution. With a pair index — a hash from each pair to the
 *   list of positions where it occurs — we only touch the O(k) positions affected
 *   by each merge and update neighbor counts in-place. The corpus is stored as a
 *   global doubly-linked list of Cells so deletions are O(1).
 *
 *   vocab_base: the first token ID to assign to merged tokens. Merge m gets ID
 *   (vocab_base + m). Ruby registers them in the same order.
 *
 * Algorithm:
 *   1. Flatten corpus into Cell array; set prev/next links per sequence.
 *   2. Scan once: build pair_counts and pair_index.
 *   3. Repeat num_merges:
 *      a. Linear scan of pair_counts for best pair (O(P), fast in C).
 *      b. For each cell_a in pair_index[best]:
 *           - Skip stale entries (cell_a.tok != tok_a or next.tok != tok_b).
 *           - L = cell_a.prev, B = cell_a.next, R = B.next.
 *           - Decrement count(L.tok, A); increment count(L.tok, new_id);
 *             push cell_L into index[(L.tok, new_id)].
 *           - Decrement count(B, R.tok); increment count(new_id, R.tok);
 *             push cell_a into index[(new_id, R.tok)].
 *           - cell_a.tok = new_id; mark cell_b DEAD; relink list.
 *      c. Zero pair_counts[best]; clear pair_index[best].
 *      d. Record [tok_a, tok_b] in merge list.
 */

#include "ruby.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── Open-addressing hash: uint64 key → int64 value ─────────────────────── */

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
    if (!pt->keys || !pt->vals) rb_raise(rb_eNoMemError, "BPE: OOM");
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
    if (!pt->keys || !pt->vals) rb_raise(rb_eNoMemError, "BPE: OOM grow");
    for (uint32_t i = 0; i < old_cap; i++) {
        if (!old_keys[i]) continue;
        uint32_t s = (uint32_t)(old_keys[i] ^ (old_keys[i] >> 17)) & (pt->cap - 1);
        while (pt->keys[s]) s = (s + 1) & (pt->cap - 1);
        pt->keys[s] = old_keys[i];
        pt->vals[s] = old_vals[i];
    }
    free(old_keys);
    free(old_vals);
}

/* Modify count for key by delta. Creates entry if absent. */
static void pt_add(PairTable *pt, uint64_t key, int64_t delta) {
    if (pt->size * 2 >= pt->cap) pt_grow(pt);
    uint32_t s = (uint32_t)(key ^ (key >> 17)) & (pt->cap - 1);
    while (pt->keys[s] && pt->keys[s] != key) s = (s + 1) & (pt->cap - 1);
    if (!pt->keys[s]) { pt->keys[s] = key; pt->size++; }
    pt->vals[s] += delta;
}

/*
 * Pack pair (a, b) as a uint64 key.
 * Token IDs are >= 0 in the corpus; we use (id+1) so 0 is the empty-slot sentinel.
 * IDs fit in 31 bits in practice (vocab < 100K).
 */
static inline uint64_t pack(int32_t a, int32_t b) {
    return ((uint64_t)(uint32_t)(a + 1) << 32) | (uint32_t)(b + 1);
}

/* ── Cell index: uint64 key → dynamic array of int32 cell indices ────────── */

typedef struct {
    int32_t *data;
    int32_t  len;
    int32_t  cap;
} CellVec;

static void cv_push(CellVec *v, int32_t ci) {
    if (v->len == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->data = realloc(v->data, (size_t)v->cap * sizeof(int32_t));
        if (!v->data) rb_raise(rb_eNoMemError, "BPE: OOM cv_push");
    }
    v->data[v->len++] = ci;
}

typedef struct {
    uint64_t *keys;
    CellVec  *vecs;  /* inline — avoids per-entry allocation */
    uint32_t  cap;
    uint32_t  size;
} IndexTable;

static void idx_init(IndexTable *it) {
    it->cap  = HT_INIT_CAP;
    it->size = 0;
    it->keys = calloc(it->cap, sizeof(uint64_t));
    it->vecs = calloc(it->cap, sizeof(CellVec));
    if (!it->keys || !it->vecs) rb_raise(rb_eNoMemError, "BPE: OOM idx_init");
}

static void idx_free(IndexTable *it) {
    for (uint32_t i = 0; i < it->cap; i++)
        if (it->keys[i]) free(it->vecs[i].data);
    free(it->keys);
    free(it->vecs);
}

static void idx_grow(IndexTable *it) {
    uint32_t  old_cap  = it->cap;
    uint64_t *old_keys = it->keys;
    CellVec  *old_vecs = it->vecs;
    it->cap  = old_cap * 2;
    it->keys = calloc(it->cap, sizeof(uint64_t));
    it->vecs = calloc(it->cap, sizeof(CellVec));
    if (!it->keys || !it->vecs) rb_raise(rb_eNoMemError, "BPE: OOM idx_grow");
    for (uint32_t i = 0; i < old_cap; i++) {
        if (!old_keys[i]) continue;
        uint32_t s = (uint32_t)(old_keys[i] ^ (old_keys[i] >> 17)) & (it->cap - 1);
        while (it->keys[s]) s = (s + 1) & (it->cap - 1);
        it->keys[s] = old_keys[i];
        it->vecs[s] = old_vecs[i];
    }
    free(old_keys);
    free(old_vecs);
}

static uint32_t idx_slot(IndexTable *it, uint64_t key) {
    uint32_t s = (uint32_t)(key ^ (key >> 17)) & (it->cap - 1);
    while (it->keys[s] && it->keys[s] != key) s = (s + 1) & (it->cap - 1);
    return s;
}

static void idx_push(IndexTable *it, uint64_t key, int32_t ci) {
    if (it->size * 2 >= it->cap) idx_grow(it);
    uint32_t s = idx_slot(it, key);
    if (!it->keys[s]) { it->keys[s] = key; it->size++; }
    cv_push(&it->vecs[s], ci);
}

static CellVec *idx_get(IndexTable *it, uint64_t key) {
    uint32_t s = idx_slot(it, key);
    return it->keys[s] ? &it->vecs[s] : NULL;
}

static void idx_clear(IndexTable *it, uint64_t key) {
    uint32_t s = idx_slot(it, key);
    if (it->keys[s]) it->vecs[s].len = 0;
}

/* ── Corpus cells ─────────────────────────────────────────────────────────── */

#define DEAD_TOK INT32_MIN

typedef struct {
    int32_t tok;   /* token ID; DEAD_TOK if merged away */
    int32_t prev;  /* global cell index of previous live cell, -1 = none */
    int32_t next;  /* global cell index of next live cell,     -1 = none */
} Cell;

/* ── train_fast ───────────────────────────────────────────────────────────── */

static VALUE rb_train_fast(VALUE self, VALUE rb_corpus, VALUE rb_num_merges, VALUE rb_vocab_base) {
    Check_Type(rb_corpus,     T_ARRAY);
    Check_Type(rb_num_merges, T_FIXNUM);
    Check_Type(rb_vocab_base, T_FIXNUM);

    int32_t num_merges = (int32_t)NUM2INT(rb_num_merges);
    int32_t vocab_base = (int32_t)NUM2INT(rb_vocab_base);
    long    n_seqs     = RARRAY_LEN(rb_corpus);

    /* 1. Flatten corpus into Cell array. */
    long total_cells = 0;
    for (long s = 0; s < n_seqs; s++)
        total_cells += RARRAY_LEN(RARRAY_AREF(rb_corpus, s));

    Cell *cells = malloc((size_t)total_cells * sizeof(Cell));
    if (!cells) rb_raise(rb_eNoMemError, "BPE: OOM cells");

    int32_t gi = 0;
    for (long s = 0; s < n_seqs; s++) {
        VALUE seq = RARRAY_AREF(rb_corpus, s);
        long  len = RARRAY_LEN(seq);
        for (long i = 0; i < len; i++, gi++) {
            cells[gi].tok  = (int32_t)NUM2INT(RARRAY_AREF(seq, i));
            cells[gi].prev = (i == 0)       ? -1 : gi - 1;
            cells[gi].next = (i == len - 1) ? -1 : gi + 1;
        }
    }

    /* 2. Build pair_counts and pair_index in one pass. */
    PairTable  counts; pt_init(&counts);
    IndexTable index;  idx_init(&index);

    for (int32_t ci = 0; ci < (int32_t)total_cells; ci++) {
        if (cells[ci].tok == DEAD_TOK) continue;
        int32_t nx = cells[ci].next;
        if (nx < 0) continue;
        uint64_t key = pack(cells[ci].tok, cells[nx].tok);
        pt_add(&counts, key, 1);
        idx_push(&index, key, ci);
    }

    /* 3. Training loop. */
    VALUE merges = rb_ary_new_capa(num_merges);

    for (int32_t m = 0; m < num_merges; m++) {
        /* a. Find best pair — linear scan of counts table. */
        uint64_t best_key   = 0;
        int64_t  best_count = 0;
        for (uint32_t i = 0; i < counts.cap; i++) {
            if (counts.keys[i] && counts.vals[i] > best_count) {
                best_count = counts.vals[i];
                best_key   = counts.keys[i];
            }
        }
        if (best_count <= 0) break;

        int32_t tok_a  = (int32_t)((best_key >> 32) & 0xFFFFFFFF) - 1;
        int32_t tok_b  = (int32_t)(best_key & 0xFFFFFFFF) - 1;
        int32_t new_id = vocab_base + m;

        /* b. Apply merge at each indexed occurrence.
         *
         * Snapshot occ->len and occ->data before the inner loop. idx_push calls
         * inside the loop can trigger idx_grow, which frees index.vecs (the array
         * of CellVec structs) and reassigns it->vecs to new memory — leaving occ
         * as a dangling pointer. The individual .data arrays are NOT freed by
         * idx_grow (they are moved via struct copy), so occ_data remains valid.
         * We never push to best_key during this loop, so occ_data is not realloced. */
        CellVec *occ = idx_get(&index, best_key);
        if (occ) {
            int32_t  occ_len  = occ->len;
            int32_t *occ_data = occ->data;
            for (int32_t oi = 0; oi < occ_len; oi++) {
                int32_t ci_a = occ_data[oi];

                /* Staleness check. */
                if (cells[ci_a].tok != tok_a) continue;
                int32_t ci_b = cells[ci_a].next;
                if (ci_b < 0 || cells[ci_b].tok != tok_b) continue;

                int32_t ci_l = cells[ci_a].prev;
                int32_t ci_r = cells[ci_b].next;

                /* Update left neighbor pair: (L, A) → (L, new_id). */
                if (ci_l >= 0 && cells[ci_l].tok != DEAD_TOK) {
                    int32_t tok_l = cells[ci_l].tok;
                    pt_add(&counts, pack(tok_l, tok_a),  -1);
                    pt_add(&counts, pack(tok_l, new_id), +1);
                    idx_push(&index, pack(tok_l, new_id), ci_l);
                }

                /* Update right neighbor pair: (B, R) → (new_id, R). */
                if (ci_r >= 0 && cells[ci_r].tok != DEAD_TOK) {
                    int32_t tok_r = cells[ci_r].tok;
                    pt_add(&counts, pack(tok_b,  tok_r), -1);
                    pt_add(&counts, pack(new_id, tok_r), +1);
                    idx_push(&index, pack(new_id, tok_r), ci_a);
                }

                /* Merge: set cell_a to new_id, delete cell_b, relink. */
                cells[ci_a].tok  = new_id;
                cells[ci_b].tok  = DEAD_TOK;
                cells[ci_a].next = ci_r;
                if (ci_r >= 0) cells[ci_r].prev = ci_a;
            }
        }

        /* c. Clear exhausted pair. */
        pt_add(&counts, best_key, -best_count);
        idx_clear(&index, best_key);

        /* d. Record merge. */
        rb_ary_push(merges, rb_ary_new_from_args(2, INT2NUM(tok_a), INT2NUM(tok_b)));

        /* Progress report. */
        int report_every = num_merges / 10;
        if (report_every < 1) report_every = 1;
        if ((m + 1) % report_every == 0 || m + 1 == num_merges) {
            int pct = (m + 1) * 100 / num_merges;
            rb_funcall(rb_stdout, rb_intern("print"), 1,
                       rb_sprintf("\r  BPE: %d/%d merges (%d%%)", m + 1, num_merges, pct));
            rb_funcall(rb_stdout, rb_intern("flush"), 0);
        }
    }

    pt_free(&counts);
    idx_free(&index);
    free(cells);

    return merges;
}

/* ── count_pairs — kept for the naive fallback path ─────────────────────── */

static VALUE rb_count_pairs(VALUE self, VALUE corpus) {
    Check_Type(corpus, T_ARRAY);
    PairTable pt; pt_init(&pt);
    long n_seqs = RARRAY_LEN(corpus);
    for (long s = 0; s < n_seqs; s++) {
        VALUE seq = RARRAY_AREF(corpus, s);
        Check_Type(seq, T_ARRAY);
        long len = RARRAY_LEN(seq);
        for (long i = 0; i < len - 1; i++) {
            long a = NUM2LONG(RARRAY_AREF(seq, i));
            long b = NUM2LONG(RARRAY_AREF(seq, i + 1));
            pt_add(&pt, pack((int32_t)a, (int32_t)b), 1);
        }
    }
    VALUE result = rb_hash_new();
    for (uint32_t i = 0; i < pt.cap; i++) {
        if (!pt.keys[i] || pt.vals[i] <= 0) continue;
        long a = (long)((pt.keys[i] >> 32) & 0xFFFFFFFF) - 1;
        long b = (long)(pt.keys[i]         & 0xFFFFFFFF) - 1;
        rb_hash_aset(result,
                     rb_ary_new_from_args(2, LONG2NUM(a), LONG2NUM(b)),
                     LONG2NUM(pt.vals[i]));
    }
    pt_free(&pt);
    return result;
}

/* ── merge_pair — single-sequence substitution ───────────────────────────── */

static VALUE rb_merge_pair(VALUE self, VALUE seq, VALUE r_a, VALUE r_b, VALUE r_new) {
    Check_Type(seq, T_ARRAY);
    long len    = RARRAY_LEN(seq);
    long pair_a = NUM2LONG(r_a);
    long pair_b = NUM2LONG(r_b);
    long new_id = NUM2LONG(r_new);
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
    rb_define_module_function(mExt, "train_fast",  rb_train_fast,  3);
}
