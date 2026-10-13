"""FP16 grouped-expert GEMM with FP32 accumulation and native output dtype."""
# Copyright 2026 Peisong Xiao (MIT).

import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["M"])
def _mm(X, W, Y, M, K: tl.constexpr, N: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile, expert = tl.program_id(0), tl.program_id(1)
    tm, tn = tile // tr.cdiv(N, BN), tile % tr.cdiv(N, BN)
    r = tm * BM + tl.arange(0, BM)
    c = tn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for start in range(tr.cdiv(K, BK)):
        k = start * BK + kk
        a = tl.load(X + expert * M * K + r[:, None] * K + k[None, :],
                    (r[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(W + expert * K * N + k[:, None] * N + c[None, :],
                    (k[:, None] < K) & (c[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + expert * M * N + r[:, None] * N + c[None, :], acc,
             (r[:, None] < M) & (c[None, :] < N))


def gemm(x, weights, out, tile=(64, 128, 64, 3)):
    b, m, k = x.shape
    n = weights.shape[-1]
    bm, bn, bk, stages = tile
    _mm[(tr.cdiv(m, bm) * tr.cdiv(n, bn), b)](
        x, weights, out, m, k, n, bm, bn, bk, num_warps=4, num_stages=stages)
