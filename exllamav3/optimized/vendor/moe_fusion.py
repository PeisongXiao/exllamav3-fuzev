"""Routed-input GEMM and gate/up fusion using reconstructed EXL3 weights.

The input addressing follows the sorted-token approach used in vLLM's fused
MoE. These kernels instead consume EXL3's existing reconstructed half weights.
"""
# Copyright 2026 Peisong Xiao (MIT).

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _tiles(M, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr):
    tile = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    first = tile // (GROUP * nn) * GROUP
    size = tl.minimum(nm - first, GROUP)
    return first + tile % (GROUP * nn) % size, tile % (GROUP * nn) // size


@tr.jit
def _half_silu(g):
    # Native activation rounds exp, addition, reciprocal and product to half.
    g = g.to(tl.float16).to(tl.float32)
    e = tl.exp(-g).to(tl.float16).to(tl.float32)
    den = (1.0 + e).to(tl.float16).to(tl.float32)
    reciprocal = tl.div_rn(1.0, den).to(tl.float16).to(tl.float32)
    return (g * reciprocal).to(tl.float16).to(tl.float32)


@tr.jit(do_not_specialize=["M"])
def _gate_up(X, TOK, WG, WU, Y, M, K: tl.constexpr, N: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             GROUP: tl.constexpr):
    tm, tn = _tiles(M, N, BM, BN, GROUP)
    expert = tl.program_id(1)
    count = M
    r = tm * BM + tl.arange(0, BM)
    c = tn * BN + tl.arange(0, BN)
    token = tl.load(TOK + expert * M + r, r < count, other=0)
    kk = tl.arange(0, BK)
    ag = tl.full((BM, BN), 0.0, tl.float32)
    au = tl.full((BM, BN), 0.0, tl.float32)
    for start in range(tr.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(X + token[:, None] * K + k[None, :],
                    (r[:, None] < count) & (k[None, :] < K), other=0)
        wi = expert * K * N + k[:, None] * N + c[None, :]
        mask = (k[:, None] < K) & (c[None, :] < N)
        g = tl.load(WG + wi, mask, other=0)
        u = tl.load(WU + wi, mask, other=0)
        ag = tl.dot(x, g, ag)
        au = tl.dot(x, u, au)
    value = _half_silu(ag) * au.to(tl.float16).to(tl.float32)
    tl.store(Y + expert * M * N + r[:, None] * N + c[None, :], value,
             (r[:, None] < count) & (c[None, :] < N))


def gate_up(x, tokens, wg, wu, out, tile=(64, 64, 64, 3), group=4):
    b, m, n = out.shape
    assert x.is_contiguous() and tokens.is_contiguous() and wg.is_contiguous() and wu.is_contiguous()
    assert out.is_contiguous() and wg.shape == wu.shape == (b, x.shape[-1], n)
    assert tokens.numel() == b * m and out.dtype == torch.float16
    bm, bn, bk, stages = tile
    grid = (tr.cdiv(m, bm) * tr.cdiv(n, bn), b)
    _gate_up[grid](
        x, tokens, wg, wu, out, m, x.shape[-1], n, bm, bn, bk, group,
        num_warps=4, num_stages=stages, enable_fp_fusion=False)
