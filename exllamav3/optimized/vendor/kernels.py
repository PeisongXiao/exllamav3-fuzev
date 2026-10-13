"""Sparse prefill attention with model-sized index selection."""
# Derived from ExLlamaV3 by turboderp (MIT). Modifications Copyright 2026 Peisong Xiao.

import torch
import triton
import triton.language as tl

from exllamav3.modules.attention_fn.triton_paged import (
    _get_h32, _qc_load_kt, _qc_load_v, _rot_h32,
    _paged_attn_decode_combine_kernel, combine_subtiles,
)

@triton.jit
def _prefix_extent(indices, extent, K_PAD: tl.constexpr, PREFIX: tl.constexpr, BUDGET: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, PREFIX)
    ix = tl.load(indices + row * K_PAD + col, col < BUDGET, -1)
    last = tl.max(tl.where(ix >= 0, col + 1, 0), 0)
    tl.store(extent + row, tl.cdiv(last, 32) * 32)


@triton.jit
def _stage_qkv(k, v, ks, vs, bt, sk, sv, length,
               PAGE: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr,
               KB: tl.constexpr, VB: tl.constexpr):
    logical = tl.program_id(0) * 32 + tl.arange(0, 32)
    head = tl.program_id(1)
    valid = logical < length
    page = tl.load(bt + logical // PAGE, valid, 0)
    physical = page * PAGE + logical % PAGE
    d = tl.arange(0, HD)
    kt = _qc_load_kt(k, ks, physical, head, d, valid, KB, KVH, HD, HD)
    vt = _qc_load_v(v, vs, physical, head, d, valid, VB, KVH, HD, HD)
    offset = (logical[:, None] * KVH + head) * HD + d[None, :]
    tl.store(sk + offset, tl.trans(kt), valid[:, None])
    tl.store(sv + offset, vt, valid[:, None])


def stage_qkv(k, v, qc, context, page_size, kvh, hd):
    length = context["length"]
    sk = torch.empty((length, kvh, hd), device=k.device, dtype=torch.float16)
    sv = torch.empty_like(sk)
    ks, vs, kb, vb = qc
    _stage_qkv[(triton.cdiv(length, 32), kvh)](
        k, v, ks, vs, context["block_table"], sk, sv, length,
        page_size, kvh, hd, kb, vb, num_warps=4,
    )
    return sk, sv


@triton.jit(do_not_specialize=["k_len", "num_pages", "splits", "split_len"])
def _sparse_kernel(
    q, k, v, bt, indices, partial_o, partial_ml, output, extent,
    k_len, num_pages, splits, split_len, ks, vs, h32,
    H: tl.constexpr, KVH: tl.constexpr, PAGE: tl.constexpr, HD: tl.constexpr,
    KP: tl.constexpr, SCALE: tl.constexpr, BH: tl.constexpr, BN: tl.constexpr,
    PAGED: tl.constexpr, QCK: tl.constexpr, QCV: tl.constexpr,
    DIRECT: tl.constexpr, CLIP: tl.constexpr, STAGED: tl.constexpr, BUDGET: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    group = H // KVH
    blocks = tl.cdiv(group, BH)
    hb = pid % blocks
    bh = pid // blocks
    batch = bh // KVH
    kh = bh - batch * KVH
    rows = tl.arange(0, BH)
    local_h = hb * BH + rows
    qh = kh * group + local_h
    valid_row = local_h < group
    d = tl.arange(0, HD)
    qbase = (batch * H + qh) * HD
    qt = tl.load(q + qbase[:, None] + d[None, :], valid_row[:, None], 0.0)
    if QCK > 0:
        qt = _rot_h32(qt, h32, BH, HD)
    start = split * split_len
    end = tl.minimum(start + split_len, k_len)
    if CLIP:
        # The incomplete tail follows the model's selected-token budget.
        prefix_end = tl.load(extent + batch)
        end = prefix_end + BN
    m = tl.full((BH,), -float("inf"), tl.float32)
    l = tl.full((BH,), 0.0, tl.float32)
    acc = tl.zeros((BH, HD), tl.float32)
    for step in range(start, end, BN):
        n0 = step
        if CLIP:
            n0 = tl.where(step < prefix_end, step, BUDGET)
        n = n0 + tl.arange(0, BN)
        idx = tl.load(indices + batch * KP + n, n < k_len, -1)
        valid_n = (idx >= 0) & (n < k_len)
        safe_idx = tl.where(valid_n, idx, 0)
        if PAGED:
            phys = tl.load(bt + batch * num_pages + safe_idx // PAGE, valid_n, 0)
            tok = phys * PAGE + safe_idx % PAGE
        else:
            tok = safe_idx
        if QCK > 0 and not STAGED:
            kt = _qc_load_kt(k, ks, tok, kh, d, valid_n, QCK, KVH, HD, HD)
        else:
            kt = tl.load(k + (tok[None, :] * KVH + kh) * HD + d[:, None], valid_n[None, :], 0.0)
        scores = tl.dot(qt, kt) * SCALE
        valid = valid_row[:, None] & valid_n[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, 1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.where(valid, tl.exp(scores - m_exp[:, None]), 0.0)
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_exp))
        l = l * alpha + tl.sum(p, 1)
        if QCV > 0 and not STAGED:
            vt = _qc_load_v(v, vs, tok, kh, d, valid_n, QCV, KVH, HD, HD)
        else:
            vt = tl.load(v + (tok[:, None] * KVH + kh) * HD + d[None, :], valid_n[:, None], 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m = m_new
    if DIRECT:
        result = acc / tl.where(l == 0.0, 1.0, l)[:, None]
        if QCV > 0:
            result = _rot_h32(result, h32, BH, HD)
        tl.store(output + qbase[:, None] + d[None, :], result, valid_row[:, None])
    else:
        base = (pid * splits + split) * BH
        tl.store(partial_o + (base + rows[:, None]) * HD + d[None, :], acc)
        tl.store(partial_ml + (base + rows) * 2, m)
        tl.store(partial_ml + (base + rows) * 2 + 1, l)


def qsa_attend(q, k, v, indices, sm_scale, block_table=None, page_size=0,
               qc=None, n_kv_heads=None):
    from .. import hooks
    from exllamav3.modules.attention_fn.qsa_triton import _get_sms
    rows, heads, hd = q.shape
    if (rows < hooks.CONFIG["prefill_min_rows"] or hd != triton.next_power_of_2(hd)
            or not all(t.is_contiguous() for t in (q, k, v, indices))):
        return hooks.ORIGINALS["qsa_attend"](
            q, k, v, indices, sm_scale, block_table, page_size, qc, n_kv_heads)
    if qc is None:
        ks, vs, kb, vb = q, q, 0, 0
        kvh, h32 = k.shape[1], q
    else:
        ks, vs, kb, vb = qc
        if (n_kv_heads is None or block_table is None
                or not ks.is_contiguous() or not vs.is_contiguous()):
            return hooks.ORIGINALS["qsa_attend"](
                q, k, v, indices, sm_scale, block_table, page_size, qc, n_kv_heads)
        kvh, h32 = n_kv_heads, _get_h32(q.device)
    if block_table is not None and (not block_table.is_contiguous() or block_table.shape[0] != rows):
        return hooks.ORIGINALS["qsa_attend"](
            q, k, v, indices, sm_scale, block_table, page_size, qc, n_kv_heads)
    context = hooks.STAGE_CONTEXT.get()
    staged = bool(hooks.CONFIG["qsa_stage"] and context is not None and qc is not None)
    if staged:
        k, v = stage_qkv(k, v, qc, context, page_size, kvh, hd)
        block_table = None
    bh, bn = 16, 32
    programs = rows * kvh * triton.cdiv(heads // kvh, bh)
    kp = indices.shape[1]
    splits = max(1, min(2 * _get_sms(q.device) // programs, triton.cdiv(kp, bn), 128))
    direct = splits == 1
    budget = context["budget"] if context is not None and kp == context["k_pad"] else 0
    clip = (splits == 1 and budget > 0
            and budget % bn == 0 and 0 < kp - budget <= bn)
    out = torch.empty_like(q)
    po = q if direct else torch.empty(programs * splits * bh * hd, device=q.device, dtype=torch.float32)
    ml = q if direct else torch.empty(programs * splits * bh * 2, device=q.device, dtype=torch.float32)
    extent = indices
    with torch.cuda.device(q.device):
        if clip:
            extent = torch.empty(rows, device=q.device, dtype=torch.int32)
            _prefix_extent[(rows,)](indices, extent, kp, triton.next_power_of_2(budget), budget)
        _sparse_kernel[(programs, splits)](
            q, k, v, block_table if block_table is not None else indices,
            indices, po, ml, out, extent, kp,
            block_table.shape[1] if block_table is not None else 0,
            splits, triton.cdiv(triton.cdiv(kp, splits), bn) * bn, ks, vs, h32,
            heads, kvh, page_size if block_table is not None else 1, hd, kp,
            float(sm_scale), bh, bn, block_table is not None, kb, vb, direct, clip, staged, budget,
            num_warps=4, num_stages=2,
        )
        if not direct:
            rs, ds = combine_subtiles(bh, hd)
            _paged_attn_decode_combine_kernel[(programs, (bh // rs) * (hd // ds))](
                po, ml, out, h32, splits, ml, QCV=vb, HAS_SINKS=False, q_len=1,
                n_q_heads=heads, n_kv_heads=kvh, head_dim=hd, HD_PAD=hd, V_DIM=hd,
                BLOCK_M=1, BLOCK_H=bh, BLOCK_ROWS=bh, ROWS_SUB=rs, D_SUB=ds,
                num_warps=4, num_stages=1,
            )
    return out
