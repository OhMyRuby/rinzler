/*
 * tensor_ext.c — Native acceleration for rinzler-tensor hot paths
 *
 * Methods on Rinzler::Tensor::TensorExt:
 *
 *   bmm(a, b)   → Numo::DFloat [batch, M, N]
 *
 * bmm — batched matrix multiply
 *
 *   Replaces the Ruby loop in Tensor#bmm that called mat_dot once per batch
 *   item. The Ruby version incurred per-iteration overhead: block invocation,
 *   numo slice allocation (NARRAY_VIEW_T), mat_dot dispatch, and the result
 *   copy-back loop. This implementation works directly on raw double* pointers
 *   and loops entirely in C.
 *
 *   Inputs must be contiguous Numo::DFloat arrays (NARRAY_DATA_T). The Ruby
 *   caller ensures this by duping any views before calling in here.
 *
 *   Forward:   C[b,m,n] = Σ_k  A[b,m,k] * B[b,k,n]
 *   Backward:  dA[b,m,k] = Σ_n dC[b,m,n] * B[b,n,k]   (dC × Bᵀ per slice)
 *              dB[b,k,n] = Σ_m A[b,k,m] * dC[b,m,n]   (Aᵀ × dC per slice)
 *   Both use the same bmm kernel — just swap/transpose the arguments.
 *
 *   Performance note: at training scale (batch=8, T=128, d_head=16) these
 *   matrices fit in L1 cache. With -O3 -march=native the compiler vectorises
 *   the inner k-loop with AVX2, giving a meaningful speedup over the numo
 *   naive dot path (which these matrices are too small to route through BLAS).
 */

#include "ruby.h"
#include "numo/narray.h"

/*
 * Ensure `obj` is a contiguous NARRAY_DATA_T. If it's a view, call Ruby's
 * #dup to get a fresh contiguous copy. We do this in C rather than requiring
 * the caller to always dup — if the array is already contiguous (the common
 * case for result tensors) we skip the allocation.
 */
static VALUE ensure_contiguous(VALUE obj) {
    narray_t *na;
    GetNArray(obj, na);
    if (na->type != NARRAY_DATA_T) {
        obj = rb_funcall(obj, rb_intern("dup"), 0);
    }
    return obj;
}

/*
 * Core batched DGEMM: C[b,m,n] += A[b,m,k] * B[b,k,n]
 * All arrays are row-major contiguous double*.
 * C is written (not accumulated) — caller zeros it via Numo::DFloat.zeros.
 */
static void dgemm_batch(const double *A, const double *B, double *C,
                        long batch, long M, long K, long N) {
    for (long b = 0; b < batch; b++) {
        const double *Ab = A + b * M * K;
        const double *Bb = B + b * K * N;
        double       *Cb = C + b * M * N;

        for (long m = 0; m < M; m++) {
            for (long n = 0; n < N; n++) {
                double sum = 0.0;
                for (long k = 0; k < K; k++) {
                    sum += Ab[m * K + k] * Bb[k * N + n];
                }
                Cb[m * N + n] = sum;
            }
        }
    }
}

/*
 * TensorExt.bmm(a, b) → Numo::DFloat
 *
 *   a: Numo::DFloat [batch, M, K]
 *   b: Numo::DFloat [batch, K, N]
 *   returns: Numo::DFloat [batch, M, N]
 */
static VALUE rb_bmm(VALUE self, VALUE rb_a, VALUE rb_b) {
    rb_a = ensure_contiguous(rb_a);
    rb_b = ensure_contiguous(rb_b);

    narray_t *na, *nb;
    GetNArray(rb_a, na);
    GetNArray(rb_b, nb);

    if (na->ndim != 3 || nb->ndim != 3)
        rb_raise(rb_eArgError, "bmm: inputs must be 3-dimensional [batch, M, K] and [batch, K, N]");

    long batch = (long)na->shape[0];
    long M     = (long)na->shape[1];
    long K     = (long)na->shape[2];
    long K2    = (long)nb->shape[1];
    long N     = (long)nb->shape[2];

    if ((long)nb->shape[0] != batch)
        rb_raise(rb_eArgError, "bmm: batch dimension mismatch (%ld vs %ld)", batch, (long)nb->shape[0]);
    if (K != K2)
        rb_raise(rb_eArgError, "bmm: inner dimension mismatch (%ld vs %ld)", K, K2);

    /* Allocate output via Ruby — Numo::DFloat.zeros(batch, M, N) */
    VALUE numo_dfloat = rb_const_get(rb_const_get(rb_cObject, rb_intern("Numo")), rb_intern("DFloat"));
    VALUE rb_c = rb_funcall(numo_dfloat, rb_intern("zeros"), 3,
                            LONG2NUM(batch), LONG2NUM(M), LONG2NUM(N));

    const double *a = (const double *)RNARRAY_DATA_PTR(rb_a);
    const double *b = (const double *)RNARRAY_DATA_PTR(rb_b);
    double       *c = (double *)RNARRAY_DATA_PTR(rb_c);

    dgemm_batch(a, b, c, batch, M, K, N);

    return rb_c;
}

void Init_tensor_ext(void) {
    VALUE mRinzler = rb_define_module("Rinzler");
    VALUE mTensor  = rb_define_module_under(mRinzler, "Tensor");
    VALUE mExt     = rb_define_module_under(mTensor, "TensorExt");

    rb_define_module_function(mExt, "bmm", rb_bmm, 2);
}
