#!/usr/bin/env python3
from __future__ import annotations
import argparse
from typing import Dict
import torch
import math

# =========================
# Helpers
# =========================

def clamp_int8_like(t: torch.Tensor) -> torch.Tensor:
    # Keep headroom math in int16, clamp to int8 range
    return t.clamp(-128, 127).to(torch.int16)

def _round_clip_int8(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x).clamp_(-128, 127).to(torch.int8)



@torch.no_grad()
def _sample_set3_match_moments_int8(
    c: Dict[str, torch.Tensor],
    per_channel: bool = False,
    seed: int = 7
) -> torch.Tensor:
    """
    Fill Set-3 positions with int8 codes sampled to match known-weight moments.
    Known = Set-1 exact (full_mean) + Set-2 midpoints ((part_min+part_max)/2), projected to int8.
    If per_channel=True, match moments per out-channel (dim=0) for Conv/Linear.
    """
    device = c["full_min"].device
    g = torch.Generator(device=device).manual_seed(seed)
    scale = float(c["scale"])
    shape = c["full_min"].shape

    q_out = torch.zeros(shape, dtype=torch.int8, device=device)

    # Masks
    mfull = c["mask_full"]
    mpart = c["mask_part"]
    mnone = c["mask_none"]
    known_mask = mfull | mpart

    if not known_mask.any():
        # Nothing known → mild fallback gaussian
        n3 = int(mnone.sum().item())
        if n3 > 0:
            samp = torch.normal(0.0, 0.5, size=(n3,), generator=g, device=device)
            q_out[mnone] = _round_clip_int8(samp)
        return q_out

    # Known int8 values: Set-1 from full_mean, Set-2 from midpoint
    q_known = torch.zeros(shape, dtype=torch.int16, device=device)
    if mfull.any():
        q_full = torch.round(c["full_mean"][mfull] / scale).clamp(-128, 127).to(torch.int16)
        q_known[mfull] = q_full
    if mpart.any():
        mids = ((c["part_min"] + c["part_max"]) * 0.5)[mpart]
        q_mid = torch.round(mids / scale).clamp(-128, 127).to(torch.int16)
        q_known[mpart] = q_mid

    # Per-tensor moments
    if not per_channel:
        qk = q_known[known_mask].to(torch.float32)
        mu  = qk.mean()
        std = qk.std(unbiased=False).clamp_min(0.5)  # prevent collapse
        p0  = (qk == 0).float().mean()

        n3 = int(mnone.sum().item())
        if n3 > 0:
            probs = torch.full((n3,), float(p0), device=device, dtype=torch.float32)
            draw_zero = torch.bernoulli(probs, generator=g).to(torch.bool)
            samp = torch.normal(mu, std, size=(n3,), generator=g, device=device)
            q3 = _round_clip_int8(samp)
            q3[draw_zero] = 0
            q_out[mnone] = q3
        return q_out

    # Per-channel moments (assume out_channels at dim=0; fall back to per-tensor if shape not suitable)
    if q_out.ndim == 0 or shape[0] == 0:
        qk = q_known[known_mask].to(torch.float32)
        mu  = qk.mean()
        std = qk.std(unbiased=False).clamp_min(0.5)
        p0  = (qk == 0).float().mean()
        n3 = int(mnone.sum().item())
        if n3 > 0:
            probs = torch.full((n3,), float(p0), device=device, dtype=torch.float32)
            draw_zero = torch.bernoulli(probs, generator=g).to(torch.bool)
            samp = torch.normal(mu, std, size=(n3,), generator=g, device=device)
            q3 = _round_clip_int8(samp)
            q3[draw_zero] = 0
            q_out[mnone] = q3
        return q_out

    out_c = shape[0]
    for oc in range(out_c):
        sel_known = known_mask[oc]
        sel_none  = mnone[oc]
        if not sel_none.any():
            continue
        if not sel_known.any():
            # No stats for this channel → mild fallback
            n3 = int(sel_none.sum().item())
            samp = torch.normal(0.0, 0.5, size=(n3,), generator=g, device=device)
            q_out[oc][sel_none] = _round_clip_int8(samp)
            continue

        qk = q_known[oc][sel_known].to(torch.float32)
        mu  = qk.mean()
        std = qk.std(unbiased=False).clamp_min(0.5)
        p0  = (qk == 0).float().mean()

        n3 = int(sel_none.sum().item())
        probs = torch.full((n3,), float(p0), device=device, dtype=torch.float32)
        draw_zero = torch.bernoulli(probs, generator=g).to(torch.bool)
        samp = torch.normal(mu, std, size=(n3,), generator=g, device=device)
        q3 = _round_clip_int8(samp)
        q3[draw_zero] = 0
        q_out[oc][sel_none] = q3

    return q_out

def _known_int8_codes_for_stats(c, scale: float) -> torch.Tensor:
    # Collect known int8 codes from Set-1 (full_mean) and Set-2 (midpoints)
    mfull, mpart = c["mask_full"], c["mask_part"]
    vals = []
    if mfull.any():
        q_full = torch.round(c["full_mean"][mfull] / scale).clamp(-128, 127)
        vals.append(q_full)
    if mpart.any():
        mids = ((c["part_min"] + c["part_max"]) * 0.5)[mpart]
        q_mid = torch.round(mids / scale).clamp(-128, 127)
        vals.append(q_mid)
    if len(vals) == 0:
        return None
    return torch.cat(vals).to(torch.int16)

def _per_channel_loc_from_known(c, scale: float, loc="mean"):
    """
    Returns a tensor of shape (out_channels,) with per-channel location (mean or median)
    computed from known int8 codes. Falls back to global if a channel has no known codes.
    """
    shape = c["full_min"].shape
    if len(shape) == 0 or shape[0] == 0:
        # scalar or empty: just global
        all_known = _known_int8_codes_for_stats(c, scale)
        if all_known is None:
            return torch.tensor(0, dtype=torch.float32, device=c["full_min"].device)
        return (all_known.float().mean() if loc == "mean" else all_known.float().median())
    out_c = shape[0]
    device = c["full_min"].device
    locs = torch.zeros(out_c, dtype=torch.float32, device=device)
    # Build a mask of known idx per channel
    mknown = c["mask_full"] | c["mask_part"]
    # Precompute int8 codes arrays to avoid recomputing per channel
    q_full = torch.zeros_like(c["full_min"], dtype=torch.int16)
    q_mid  = torch.zeros_like(c["full_min"], dtype=torch.int16)
    if c["mask_full"].any():
        q_full[c["mask_full"]] = torch.round(c["full_mean"][c["mask_full"]]/scale).clamp(-128,127).to(torch.int16)
    if c["mask_part"].any():
        mids = ((c["part_min"] + c["part_max"]) * 0.5)
        q_mid[c["mask_part"]]  = torch.round(mids[c["mask_part"]]/scale).clamp(-128,127).to(torch.int16)
    for oc in range(out_c):
        sel = mknown[oc]
        if sel.any():
            data = torch.cat([q_full[oc][sel], q_mid[oc][sel]]).to(torch.float32)
            locs[oc] = (data.mean() if loc == "mean" else data.median())
        else:
            locs[oc] = 0.0
    return locs

"""
Implement closer-to-zero reconstruction (CZR) for partially recovered INT8 weights.

This module provides a function ``czr_int8_from_constraints`` which reconstructs a
quantized weight tensor from leaked float bounds.  The reconstruction follows the
algorithm described by Ghavami et al. [ICCD 2024]:

* For each parameter with a **known sign bit**, all unknown bits are filled in
  so that the resulting two's-complement value is as close to zero as possible.
  Concretely, if the sign bit is 0 (non-negative) we set unknown bits to 0,
  producing the smallest possible non-negative code.  If the sign bit is 1
  (negative), we set unknown bits to 1, producing the largest possible
  negative code (-1).
* For parameters with an **unknown sign**, we evaluate both possibilities.  We
  construct one candidate where the sign bit is 0 and all unknown bits are set
  to 0 (yielding 0), and another candidate where the sign bit is 1 and all
  unknown bits are set to 1 (yielding -1).  The candidate with the smaller
  absolute value is chosen as the reconstruction.

The function operates directly in the INT8 domain: it accepts per-tensor
constraints describing the minimum and maximum float values at each position
(`part_min` and `part_max`), a binary mask selecting which values fall into
Set-2 (`mask_part`), and a scalar scale factor used for quantization.  It
returns an INT8 tensor where Set-2 positions have been reconstructed according
to CZR and all other positions are untouched.

This code is meant as a drop-in replacement for the Set-2 branch of the
``czr_int8_from_constraints`` function in the original script.  You can call
``reconstruct_set2`` from within your existing implementation to obtain the
Set-2 reconstruction.
"""



def reconstruct_set2(
    part_min: torch.Tensor,
    part_max: torch.Tensor,
    mask_part: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reconstruct Set-2 parameters according to the CZR algorithm.

    Parameters
    ----------
    part_min, part_max : torch.Tensor
        Per-element lower and upper bounds (float) on the original weight,
        recovered from leakage.  Only the positions where ``mask_part`` is
        ``True`` will be reconstructed; all other positions are left as zero.

    mask_part : torch.Tensor (bool)
        Boolean mask identifying Set-2 positions.  ``True`` where the weight
        has been partially recovered and needs reconstruction.

    scale : float
        The quantization scale.  Used to convert float bounds into int8
        integer ranges via ``q = round(w / scale)``.

    Returns
    -------
    torch.Tensor (int8)
        An INT8 tensor containing the reconstructed values at the Set-2
        positions.  All positions not selected by ``mask_part`` are zero.

    Notes
    -----
    The reconstruction follows a sign-aware strategy.  We first compute
    conservative integer bounds ``qmin`` and ``qmax`` by dividing the
    float bounds by ``scale`` and taking the ceiling/floor【148828402571559†L815-L830】.
    If ``qmin >= 0``, the sign bit is known to be 0.  In that case we
    choose ``qmin`` (the smallest non-negative code), corresponding to
    setting all unknown bits to 0.  If ``qmax <= -1``, the sign bit is
    known to be 1.  Then we choose ``qmax`` (the largest negative code),
    corresponding to setting all unknown bits to 1.  Otherwise the sign is
    ambiguous: we test the canonical positive candidate ``0`` and the
    canonical negative candidate ``-1`` and pick whichever is legal and
    closer to zero【148828402571559†L823-L831】.
    """

    device = part_min.device
    # Prepare output buffer.  Initialise as zeros everywhere.
    q_out = torch.zeros_like(part_min, dtype=torch.int8, device=device)

    # Convert float bounds into conservative integer bounds in the int8 domain.
    # We clamp to the int8 range [−128, 127] after rounding.
    qmin = torch.ceil(part_min / scale).to(torch.int16).clamp_(-128, 127)
    qmax = torch.floor(part_max / scale).to(torch.int16).clamp_(-128, 127)

    # Case 1: sign bit is 0 (non‑negative): pick the smallest non‑negative code.
    pos_mask = mask_part & (qmin >= 0)
    if pos_mask.any():
        q_out[pos_mask] = qmin[pos_mask].to(torch.int8)

    # Case 2: sign bit is 1 (negative): pick the largest negative code.
    neg_mask = mask_part & (qmax <= -1)
    if neg_mask.any():
        q_out[neg_mask] = qmax[neg_mask].to(torch.int8)

    # Case 3: sign ambiguous – test both possibilities and pick the one with
    # smaller magnitude.  Only positions where neither pos_mask nor neg_mask
    # apply need this logic.
    amb_mask = mask_part & ~(pos_mask | neg_mask)
    if amb_mask.any():
        qm = qmin[amb_mask]
        qM = qmax[amb_mask]
        # Candidate positive: 0 (all bits zero).
        cand_pos = torch.zeros_like(qm, dtype=torch.int16, device=device)
        # Candidate negative: -1 (all bits one).
        cand_neg = torch.full_like(qm, -1, dtype=torch.int16, device=device)
        # Check if candidates lie within the feasible interval [qm, qM].
        pos_ok = (cand_pos >= qm) & (cand_pos <= qM)
        neg_ok = (cand_neg >= qm) & (cand_neg <= qM)
        # Initialise chosen candidate with sentinel value; will be filled below.
        chosen = torch.empty_like(qm, dtype=torch.int16, device=device)
        # Where a zero candidate is valid, pick zero – it minimises |q|.
        chosen[pos_ok] = 0
        # Where zero is invalid but −1 is valid, pick −1.
        only_neg = (~pos_ok) & neg_ok
        chosen[only_neg] = -1
        # Where neither 0 nor −1 are within bounds, fall back to the
        # endpoint with the smaller magnitude.  Compare |qm| vs |qM|.
        neither = (~pos_ok) & (~neg_ok)
        if neither.any():
            qm_nei = qm[neither]
            qM_nei = qM[neither]
            pick_qm = qm_nei.abs() <= qM_nei.abs()
            chosen[neither] = torch.where(pick_qm, qm_nei, qM_nei)
        # Write the chosen values into the output tensor.
        q_out[amb_mask] = chosen.to(torch.int8)

    return q_out


def czr_int8_from_constraints(
    c: Dict[str, torch.Tensor],
    fallback_strategy: str = "keep_orig",
    orig_q: torch.Tensor | None = None,
    seed: int = 7,
    set2_strategy: str = "czr",  # NEW: how to handle Set-2
) -> torch.Tensor:
    """Wrapper around CZR that applies Set-1 and Set-3 logic from the original script.

    This function performs the full reconstruction pipeline:
    1. **Set-1**: positions where the float mean is exactly known are quantised via
       rounding and clamping.
    2. **Set-2**: positions where only float bounds are known are reconstructed
       using ``reconstruct_set2`` described above.
    3. **Set-3**: positions with no information fall back to a user-selectable
       strategy (e.g., keep the original int8 codes, fill with zeros, sample
       Gaussian noise, match moments, etc.).

    Parameters
    ----------
    c : Dict[str, torch.Tensor]
        A dictionary containing the following keys:
        ``full_min``, ``full_max``, ``full_mean`` (float tensors for Set-1),
        ``part_min``, ``part_max`` (float tensors for Set-2),
        ``mask_full``, ``mask_part``, ``mask_none`` (boolean masks), and
        ``scale`` (a float scalar).  The structure matches the leakage
        constraints format used in the original script.

    fallback_strategy : str, optional
        Strategy to use for Set-3 values.  Supported options are: "keep_orig",
        "zeros", "gaussian", "match_moments", "match_moments_per_channel",
        "map_gaussian", "map_laplace".  See the original implementation for
        details on these strategies.  Default is "keep_orig".

    orig_q : torch.Tensor, optional
        Original int8 codes for the parameter.  Required when
        ``fallback_strategy`` is "keep_orig".  Ignored otherwise.

    seed : int, optional
        Random seed for stochastic fallbacks.  Default is 7.

    Returns
    -------
    torch.Tensor (int8)
        A reconstructed int8 tensor with the same shape as ``full_min``.
    """

    device = c["full_min"].device
    scale = float(c["scale"])
    shape = c["full_min"].shape

    # Initialise output tensor.
    q_out = torch.zeros(shape, dtype=torch.int8, device=device)

    # ----- Set‑1: positions with exact means.
    mask_full = c.get("mask_full")
    if mask_full is not None and mask_full.any():
        full_mean = c["full_mean"]
        q_full = torch.round(full_mean / scale).clamp(-128, 127).to(torch.int8)
        q_out[mask_full] = q_full[mask_full]

    # ----- Set-2: positions with float bounds.
    mask_part = c.get("mask_part")
    if mask_part is not None and mask_part.any():
        part_min = c["part_min"]
        part_max = c["part_max"]

        if set2_strategy == "czr":
            # Your current closer-to-zero reconstruction (B3FA-style).
            q_set2 = reconstruct_set2(part_min, part_max, mask_part, scale)

        elif set2_strategy == "deepsteal_mid":
            # DeepSteal-style init: take the midpoint of the projected range
            # W_mean = (W_min + W_max)/2, then quantise.
            mid = (part_min + part_max) * 0.5
            q_set2 = torch.round(mid / scale).clamp(-128, 127).to(torch.int8)

        else:
            raise ValueError(f"Unknown set2_strategy: {set2_strategy}")

        q_out[mask_part] = q_set2[mask_part]

    # ----- Set‑3: positions with no information – apply fallback.
    mask_none = c.get("mask_none")
    if mask_none is not None and mask_none.any():
        if fallback_strategy == "keep_orig":
            if orig_q is None:
                raise ValueError("orig_q must be provided when fallback_strategy == 'keep_orig'.")
            q_out[mask_none] = orig_q[mask_none]
        elif fallback_strategy == "zeros":
            q_out[mask_none] = 0
        elif fallback_strategy == "gaussian":
            # Estimate standard deviation from known values.  If no known
            # positions exist, use a small default.
            mids = []
            if mask_full.any():
                mids.append(c["full_mean"][mask_full])
            if mask_part.any():
                mids.append(((c["part_min"] + c["part_max"]) * 0.5)[mask_part])
            if len(mids) > 0:
                mids_cat = torch.cat(mids).to(torch.float32)
                std = mids_cat.std().item() if mids_cat.numel() > 1 else 0.01
            else:
                std = 0.01
            g = torch.Generator(device=device).manual_seed(seed)
            samp = torch.normal(mean=0.0, std=std, size=(int(mask_none.sum().item()),), generator=g, device=device)
            q_none = torch.round(samp / scale).clamp(-128, 127).to(torch.int8)
            q_out[mask_none] = q_none
        elif fallback_strategy == "match_moments":
            # Sample Set‑3 values to match the global mean/std and probability of zero
            # of the known int8 codes.  See original implementation for details.
            q_mm = _sample_set3_match_moments_int8(c, per_channel=False, seed=seed)
            q_out[mask_none] = q_mm[mask_none]
        elif fallback_strategy == "match_moments_per_channel":
            q_mm = _sample_set3_match_moments_int8(c, per_channel=True, seed=seed)
            q_out[mask_none] = q_mm[mask_none]
        elif fallback_strategy == "map_gaussian":
            # Fill with per‑channel mean of known int8 codes.
            locs = _per_channel_loc_from_known(c, scale, loc="mean")
            if locs.ndim == 0:
                q_fill = torch.round(locs).clamp(-128, 127).to(torch.int8)
                q_out[mask_none] = q_fill
            else:
                q_fill = torch.round(locs).clamp(-128, 127).to(torch.int8)
                expand_shape = (q_fill.shape[0],) + (1,) * max(q_out.ndim - 1, 0)
                q_base = q_fill.reshape(expand_shape)
                q_out[mask_none] = q_base.expand_as(q_out)[mask_none]
        elif fallback_strategy == "map_laplace":
            # --- Discrete MAP under Laplace prior (per-channel) ---
            device = c["full_min"].device
            scale = float(c["scale"])
            shape = c["full_min"].shape
            # Collect known int8 codes per channel (as in your helpers)
            mfull, mpart, mnone = c["mask_full"], c["mask_part"], c["mask_none"]

            # Build int16 known code tensor aligned with shape
            q_known = torch.zeros(shape, dtype=torch.int16, device=device)
            if mfull.any():
                q_known[mfull] = torch.round(c["full_mean"][mfull] / scale).clamp(-128,127).to(torch.int16)
            if mpart.any():
                mids = (c["part_min"] + c["part_max"]) * 0.5
                q_known[mpart] = torch.round(mids[mpart] / scale).clamp(-128,127).to(torch.int16)

            # Candidate code grid (full int8 range)
            codes = torch.arange(-128, 128, dtype=torch.int16, device=device).to(torch.float32)

            def fit_laplace_codes(qc: torch.Tensor):
                # Fit Laplace on codes directly (robust): mu = median, b = mean(|x-mu|)
                mu = qc.median()
                b  = (qc - mu).abs().mean().clamp_min(1e-6)
                return mu.item(), b.item()

            # Per-channel; if shape[0] missing/0, fall back to tensor-level
            if q_out.ndim == 0 or shape[0] == 0:
                known_vec = q_known[(mfull | mpart)]
                if known_vec.numel() == 0:
                    q_out[mnone] = 0
                else:
                    mu, b = fit_laplace_codes(known_vec.to(torch.float32))
                    # Optional zero snap in code space
                    if abs(mu) < 0.25:  # ~quarter-code tolerance
                        mu = 0.0
                    logp = - (codes - mu).abs() / b - math.log(2.0 * b)
                    best_code = codes[torch.argmax(logp)].round().clamp(-128,127).to(torch.int8)
                    q_out[mnone] = best_code
            else:
                OC = shape[0]
                for oc in range(OC):
                    sel_known = (mfull | mpart)[oc]
                    sel_none  = mnone[oc]
                    if not sel_none.any():
                        continue
                    if not sel_known.any():
                        q_out[oc][sel_none] = 0
                        continue
                    qc = q_known[oc][sel_known].to(torch.float32)
                    mu, b = fit_laplace_codes(qc)
                    if abs(mu) < 0.25:
                        mu = 0.0
                    logp = - (codes - mu).abs() / b - math.log(2.0 * b)
                    best_code = codes[torch.argmax(logp)].round().clamp(-128,127).to(torch.int8)
                    q_out[oc][sel_none] = best_code
        else:
            raise ValueError(f"Unknown fallback_strategy: {fallback_strategy}")

    return q_out


@torch.no_grad()
def _sample_set3_match_moments_int8(
    c: Dict[str, torch.Tensor],
    per_channel: bool = False,
    seed: int = 7,
) -> torch.Tensor:
    """Sample Set-3 int8 values to match known moments.

    This helper function retains the behaviour of the original script.  It
    computes global or per-channel statistics of the known int8 codes (mean,
    standard deviation, and probability of zero), then draws samples from a
    Gaussian distribution in the int8 domain and enforces a zero-probability
    exactly by Bernoulli sampling.  It is provided here verbatim for
    completeness; see the original implementation for derivation.
    """
    device = c["full_min"].device
    g = torch.Generator(device=device).manual_seed(seed)
    scale = float(c["scale"])
    shape = c["full_min"].shape
    q_out = torch.zeros(shape, dtype=torch.int8, device=device)
    mfull = c["mask_full"]
    mpart = c["mask_part"]
    mnone = c["mask_none"]
    known_mask = mfull | mpart
    # If nothing is known, fall back to a mild Gaussian.
    if not known_mask.any():
        n3 = int(mnone.sum().item())
        if n3 > 0:
            samp = torch.normal(0.0, 0.5, size=(n3,), generator=g, device=device)
            q_out[mnone] = torch.round(samp).clamp(-128, 127).to(torch.int8)
        return q_out
    # Build an int16 tensor of known int8 codes.
    q_known = torch.zeros(shape, dtype=torch.int16, device=device)
    if mfull.any():
        q_full = torch.round(c["full_mean"][mfull] / scale).clamp(-128, 127).to(torch.int16)
        q_known[mfull] = q_full
    if mpart.any():
        mids = ((c["part_min"] + c["part_max"]) * 0.5)[mpart]
        q_mid = torch.round(mids / scale).clamp(-128, 127).to(torch.int16)
        q_known[mpart] = q_mid
    # Global moments.
    if not per_channel:
        qk = q_known[known_mask].to(torch.float32)
        mu = qk.mean()
        std = qk.std(unbiased=False).clamp_min(0.5)
        p0 = (qk == 0).float().mean()
        n3 = int(mnone.sum().item())
        if n3 > 0:
            probs = torch.full((n3,), float(p0), device=device, dtype=torch.float32)
            draw_zero = torch.bernoulli(probs, generator=g).to(torch.bool)
            samp = torch.normal(mu, std, size=(n3,), generator=g, device=device)
            q3 = torch.round(samp).clamp(-128, 127).to(torch.int8)
            q3[draw_zero] = 0
            q_out[mnone] = q3
        return q_out
    # Per‑channel moments.
    if q_out.ndim == 0 or shape[0] == 0:
        # Scalar or empty: fall back to global.
        qk = q_known[known_mask].to(torch.float32)
        mu = qk.mean()
        std = qk.std(unbiased=False).clamp_min(0.5)
        p0 = (qk == 0).float().mean()
        n3 = int(mnone.sum().item())
        if n3 > 0:
            probs = torch.full((n3,), float(p0), device=device, dtype=torch.float32)
            draw_zero = torch.bernoulli(probs, generator=g).to(torch.bool)
            samp = torch.normal(mu, std, size=(n3,), generator=g, device=device)
            q3 = torch.round(samp).clamp(-128, 127).to(torch.int8)
            q3[draw_zero] = 0
            q_out[mnone] = q3
        return q_out
    out_c = shape[0]
    for oc in range(out_c):
        sel_known = known_mask[oc]
        sel_none = mnone[oc]
        if not sel_none.any():
            continue
        if not sel_known.any():
            n3 = int(sel_none.sum().item())
            samp = torch.normal(0.0, 0.5, size=(n3,), generator=g, device=device)
            q_out[oc][sel_none] = torch.round(samp).clamp(-128, 127).to(torch.int8)
            continue
        qk = q_known[oc][sel_known].to(torch.float32)
        mu = qk.mean()
        std = qk.std(unbiased=False).clamp_min(0.5)
        p0 = (qk == 0).float().mean()
        n3 = int(sel_none.sum().item())
        probs = torch.full((n3,), float(p0), device=device, dtype=torch.float32)
        draw_zero = torch.bernoulli(probs, generator=g).to(torch.bool)
        samp = torch.normal(mu, std, size=(n3,), generator=g, device=device)
        q3 = torch.round(samp).clamp(-128, 127).to(torch.int8)
        q3[draw_zero] = 0
        q_out[oc][sel_none] = q3
    return q_out


def _per_channel_loc_from_known(c: Dict[str, torch.Tensor], scale: float, loc: str = "mean") -> torch.Tensor:
    """Compute per-channel mean or median of known int8 codes.

    This helper replicates the corresponding function from the original script.
    It returns a 1D tensor of length equal to the number of output channels
    containing the per-channel location statistic (mean or median) of the
    known int8 codes.  If no values are known for a channel, it falls back
    to zero.  The returned tensor lives on the same device as ``c['full_min']``.
    """
    shape = c["full_min"].shape
    device = c["full_min"].device
    if len(shape) == 0 or shape[0] == 0:
        all_known = _known_int8_codes_for_stats(c, scale)
        if all_known is None:
            return torch.tensor(0, dtype=torch.float32, device=device)
        return all_known.float().mean() if loc == "mean" else all_known.float().median()
    out_c = shape[0]
    locs = torch.zeros(out_c, dtype=torch.float32, device=device)
    mknown = c["mask_full"] | c["mask_part"]
    q_full = torch.zeros_like(c["full_min"], dtype=torch.int16)
    q_mid = torch.zeros_like(c["full_min"], dtype=torch.int16)
    if c["mask_full"].any():
        q_full[c["mask_full"]] = torch.round(c["full_mean"][c["mask_full"]] / scale).clamp(-128, 127).to(torch.int16)
    if c["mask_part"].any():
        mids = (c["part_min"] + c["part_max"]) * 0.5
        q_mid[c["mask_part"]] = torch.round(mids[c["mask_part"]] / scale).clamp(-128, 127).to(torch.int16)
    for oc in range(out_c):
        sel = mknown[oc]
        if sel.any():
            data = torch.cat([q_full[oc][sel], q_mid[oc][sel]]).to(torch.float32)
            locs[oc] = data.mean() if loc == "mean" else data.median()
        else:
            locs[oc] = 0.0
    return locs


def _known_int8_codes_for_stats(c: Dict[str, torch.Tensor], scale: float) -> torch.Tensor | None:
    """Collect known int8 codes from Set‑1 and Set‑2 for statistics.
    Returns ``None`` if no codes are known."""
    mfull, mpart = c["mask_full"], c["mask_part"]
    vals = []
    if mfull.any():
        q_full = torch.round(c["full_mean"][mfull] / scale).clamp(-128, 127)
        vals.append(q_full)
    if mpart.any():
        mids = ((c["part_min"] + c["part_max"]) * 0.5)[mpart]
        q_mid = torch.round(mids / scale).clamp(-128, 127)
        vals.append(q_mid)
    if len(vals) == 0:
        return None
    return torch.cat(vals).to(torch.int16)


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim_int8", default="victim_vgg16_cifar10_int8.pt",
                    help="Path to int8-packed victim weights (.pt)")
    ap.add_argument("--constraints", default="leak_constraints.pt",
                    help="Path to leak constraints (.pt)")
    ap.add_argument("--out", default="reconstructed_czr_int8.pt",
                    help="Output path for reconstructed int8-packed weights")
    ap.add_argument("--fallback_none", default="keep_orig",
                    choices=["keep_orig", "zeros", "gaussian", "match_moments", "match_moments_per_channel", "map_gaussian", "map_laplace"],
                    help="Strategy for Set-3 weights")
    ap.add_argument(
        "--set2_strategy",
        default="czr",
        choices=["czr", "deepsteal_mid"],
        help="How to reconstruct Set-2 weights: "
             "'czr' = closer-to-zero (B3FA), "
             "'deepsteal_mid' = quantised midpoint as in DeepSteal init."
    )
    ap.add_argument("--seed", type=int, default=7, help="RNG seed for stochastic fallbacks")
    args = ap.parse_args()

    # Load int8 victim and constraints
    victim_pack: Dict[str, torch.Tensor] = torch.load(args.victim_int8, map_location="cpu")
    cons_pack = torch.load(args.constraints, map_location="cpu")
    constraints: Dict[str, Dict[str, torch.Tensor]] = cons_pack["constraints"]

    recon_pack: Dict[str, torch.Tensor] = {}
    used = set()

    SUFFIX = "::qint8"
    for k, v in victim_pack.items():
        if not (isinstance(v, torch.Tensor) and k.endswith(SUFFIX)):
            # copy non-q tensors (e.g., scales, buffers) unchanged
            recon_pack[k] = v.clone() if isinstance(v, torch.Tensor) else v
            continue

        base = k[:-len(SUFFIX)]
        if base in constraints:
            c = constraints[base]
            orig_q = v if args.fallback_none == "keep_orig" else None
            q_recon = czr_int8_from_constraints(
                c,
                fallback_strategy=args.fallback_none,
                orig_q=orig_q,
                seed=args.seed,
                set2_strategy=args.set2_strategy,  # NEW
            )
            recon_pack[k] = q_recon.cpu()
            # keep original scale tensor for that param
            scale_key = base + "::scale"
            if scale_key in victim_pack:
                recon_pack[scale_key] = victim_pack[scale_key].clone()
            used.add(base)
        else:
            # no constraints → keep original tensor (and its scale, if any)
            recon_pack[k] = v.clone()
            scale_key = base + "::scale"
            if scale_key in victim_pack:
                recon_pack[scale_key] = victim_pack[scale_key].clone()

    print(f"[INFO] used constraints for {len(used)} tensors out of "
          f"{sum(1 for kk in victim_pack if isinstance(victim_pack[kk], torch.Tensor) and kk.endswith(SUFFIX))}")
    torch.save(recon_pack, args.out)
    print(f"[OK] Wrote CZR-reconstructed int8-packed weights to {args.out}")

if __name__ == "__main__":
    main()
