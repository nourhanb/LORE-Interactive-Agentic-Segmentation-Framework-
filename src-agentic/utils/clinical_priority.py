"""
Clinical priority ranking of FN/FP error regions (no organ masks required).

priority(c) = v_c * (1 + λ_iso * isolation + λ_int * intensity + λ_bnd * boundary)

Used to steer heatmap supervision and simulated physician prompts toward the
error component that matters most clinically when OAR maps are unavailable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from scipy import ndimage


def _pred_centroid(pred_bin: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Centroid of binary prediction; None if empty."""
    if not pred_bin.any():
        return None
    return tuple(float(x) for x in ndimage.center_of_mass(pred_bin))


def _boundary_mask(pred_bin: np.ndarray) -> np.ndarray:
    """1-voxel morphological boundary of the prediction."""
    if not pred_bin.any():
        return np.zeros_like(pred_bin, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(pred_bin, structure=structure)
    return pred_bin & ~eroded


def select_priority_component(
    delta_union: np.ndarray,
    pred_bin: np.ndarray,
    image: Optional[np.ndarray] = None,
    lam_iso: float = 1.0,
    lam_int: float = 0.5,
    lam_bnd: float = 0.5,
    max_components: int = 32,
) -> Tuple[Optional[int], Optional[Tuple[float, float, float]], Optional[np.ndarray], float]:
    """
    Rank connected components of FN∪FP by clinical priority.

    Only the ``max_components`` largest CCs (by volume) are scored — curriculum
    noise can otherwise create thousands of speckles and stall ranking.

    Returns
    -------
    best_label : int or None
        Label id in `ndimage.label` output (1-indexed).
    centroid : (x,y,z) or None
    priority_mask : bool ndarray of the winning CC, or None
    priority_score : float
        Priority of the winning CC (0 if none).
    """
    delta_union = delta_union.astype(bool)
    if not delta_union.any():
        return None, None, None, 0.0

    labeled, n_comp = ndimage.label(delta_union)
    if n_comp == 0:
        return None, None, None, 0.0

    pred_bin = pred_bin.astype(bool)
    pred_c = _pred_centroid(pred_bin)
    bnd = _boundary_mask(pred_bin)
    total_err = float(delta_union.sum()) + 1e-8
    # Max possible distance in the volume (diagonal) for isolation normalisation
    X, Y, Z = delta_union.shape
    diag = float(np.sqrt(X * X + Y * Y + Z * Z) + 1e-8)

    pred_mean_int = None
    if image is not None and pred_bin.any():
        pred_mean_int = float(image[pred_bin].mean())

    # Restrict scoring to the largest CCs by volume (avoids noise speckles)
    sizes = ndimage.sum(delta_union, labeled, range(1, n_comp + 1))
    if np.isscalar(sizes):
        sizes = np.array([sizes])
    else:
        sizes = np.asarray(sizes)
    order = np.argsort(sizes)[::-1][:max(1, min(max_components, n_comp))]
    candidate_labs = (order + 1).tolist()

    best_label = int(candidate_labs[0])
    best_score = -1.0
    best_centroid = None
    best_mask = None

    for lab in candidate_labs:
        mask = labeled == lab
        vol = float(mask.sum())
        v_c = vol / total_err

        centroid = ndimage.center_of_mass(mask)
        if pred_c is None:
            isolation = 1.0  # no prediction yet → treat all as detached
        else:
            dist = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(centroid, pred_c))))
            isolation = min(1.0, dist / diag)

        if image is not None and pred_mean_int is not None and mask.any():
            cc_mean = float(image[mask].mean())
            # Soft saturation of intensity contrast
            intensity = min(1.0, abs(cc_mean - pred_mean_int) / (abs(pred_mean_int) + 1.0))
        else:
            intensity = 0.0

        boundary = float((mask & bnd).sum()) / (vol + 1e-8)

        score = v_c * (1.0 + lam_iso * isolation + lam_int * intensity + lam_bnd * boundary)
        if score > best_score:
            best_score = score
            best_label = int(lab)
            best_centroid = tuple(float(x) for x in centroid)
            best_mask = mask

    return best_label, best_centroid, best_mask, float(best_score)


def priority_mask_from_errors(
    fn_mask: torch.Tensor,   # (1,X,Y,Z) or (X,Y,Z)
    fp_mask: torch.Tensor,
    pred: torch.Tensor,      # soft or hard pred, same spatial shape
    image: Optional[torch.Tensor] = None,
    lam_iso: float = 1.0,
    lam_int: float = 0.5,
    lam_bnd: float = 0.5,
) -> Tuple[torch.Tensor, float, Optional[Tuple[float, float, float]]]:
    """
    Build a binary priority mask (same device/dtype as fn_mask) for one sample.

    Returns (priority_mask, priority_score, centroid).
    """
    def _np3(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().float().cpu().numpy()
        while arr.ndim > 3:
            arr = arr[0]
        return arr

    fn = _np3(fn_mask)
    fp = _np3(fp_mask)
    pred_bin = _np3(pred) > 0.5
    img_np = _np3(image) if image is not None else None

    delta_union = (fn + fp) > 0
    _, centroid, best_mask, score = select_priority_component(
        delta_union, pred_bin, img_np, lam_iso=lam_iso, lam_int=lam_int, lam_bnd=lam_bnd,
    )

    out = torch.zeros_like(fn_mask, dtype=torch.float32)
    if best_mask is None:
        return out, 0.0, None

    # Broadcast numpy mask into torch tensor shape
    mask_t = torch.from_numpy(best_mask.astype(np.float32))
    while mask_t.dim() < fn_mask.dim():
        mask_t = mask_t.unsqueeze(0)
    out = mask_t.to(device=fn_mask.device, dtype=torch.float32)
    return out, score, centroid


def priority_hit_score(
    corr_mask: torch.Tensor,       # (B,1,X,Y,Z) or (1,X,Y,Z) — current FN∪FP
    priority_mask: torch.Tensor,   # same spatial — high-priority CC at this step
) -> float:
    """Fraction of the correction that falls inside the priority region."""
    corr = corr_mask.float()
    prio = priority_mask.float()
    denom = corr.sum().item() + 1e-8
    return float((corr * prio).sum().item() / denom)
