"""
MEDSA — Mask-Edit Driven Spatial Agent
=======================================
Components:
  EditDeltaEncoder         — 3D CNN encodes geometric structure of physician mask edits.
  CorrectionPatternMemory  — Transformer-based session memory over past edit embeddings.
  compute_geometric_descriptors — Volume, asymmetry, compactness, boundary-overlap scalars.
  apply_curriculum_noise   — Simulates imprecise annotators by adding boundary noise to deltas.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# Geometric descriptor helpers
# ─────────────────────────────────────────────────────────────────────────────

def _surface_voxels(mask: torch.Tensor) -> torch.Tensor:
    """Count surface voxels of a binary 3-D mask using max-pooling dilation."""
    # mask: (B, 1, X, Y, Z) float
    dilated = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
    return (dilated - mask).clamp(0, 1)


def _to_5d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is (B,1,X,Y,Z). Handles MONAI MetaTensor 4-D edge-case."""
    t = t.float()
    if t.dim() == 4:
        return t.unsqueeze(1)
    if t.dim() == 5 and t.size(1) != 1:
        return t[:, :1, ...]
    return t


@torch.no_grad()
def compute_geometric_descriptors(
    delta_pos: torch.Tensor,   # (B,1,X,Y,Z) Δm+ — voxels physician added
    delta_neg: torch.Tensor,   # (B,1,X,Y,Z) Δm- — voxels physician erased
    prev_pred:  torch.Tensor,  # (B,1,X,Y,Z) y'_{i-1} binary
) -> torch.Tensor:             # (B, 4) descriptors
    """
    Returns a (B, 4) float tensor with columns:
      [edit_volume, edit_asymmetry, edit_compactness, edit_boundary_overlap]
    All values are dimensionless and in a bounded range for stability.
    """
    delta_pos = _to_5d(delta_pos)
    delta_neg = _to_5d(delta_neg)
    prev_pred = _to_5d(prev_pred)
    B = delta_pos.size(0)
    total_vox = float(delta_pos[0].numel())
    eps = 1e-6

    descriptors = []
    for b in range(B):
        dp = delta_pos[b, 0].float()   # (X,Y,Z)
        dn = delta_neg[b, 0].float()
        pp = prev_pred[b, 0].float()

        vol_pos = dp.sum()
        vol_neg = dn.sum()

        # ── edit_volume ─────────────────────────────────────
        edit_vol = (vol_pos + vol_neg) / total_vox

        # ── edit_asymmetry ───────────────────────────────────
        edit_asym = (vol_pos - vol_neg) / (vol_pos + vol_neg + eps)

        # ── edit_compactness (isoperimetric ratio) ───────────
        # Use the union delta Δm = Δm+ ∪ Δm-
        delta_union = ((dp + dn) > 0).float().unsqueeze(0).unsqueeze(0)
        surf = _surface_voxels(delta_union).sum()
        vol  = delta_union.sum()
        if vol > 0:
            # normalise: for a perfect sphere comp ≈ π^(1/3) * (6)^(2/3) ≈ 4.84
            compactness = (surf ** 1.5) / (vol + eps)
            compactness = compactness / 50.0   # soft normalise to ~[0,1]
        else:
            compactness = torch.zeros(1, device=dp.device)

        # ── edit_boundary_overlap ────────────────────────────
        # Boundary of previous prediction
        prev_surf = _surface_voxels(pp.unsqueeze(0).unsqueeze(0)).squeeze()
        boundary_vol = prev_surf.sum()
        if boundary_vol > 0:
            bnd_overlap = ((prev_surf * (dp + dn).clamp(0, 1)) > 0).float().sum() / (boundary_vol + eps)
        else:
            bnd_overlap = torch.zeros(1, device=dp.device)

        descriptors.append(torch.stack([
            edit_vol.squeeze(),
            edit_asym.squeeze(),
            compactness.squeeze(),
            bnd_overlap.squeeze(),
        ]))

    return torch.stack(descriptors, dim=0).to(delta_pos.device)   # (B, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum noise on edit deltas
# ─────────────────────────────────────────────────────────────────────────────

def apply_curriculum_noise(
    delta_pos: torch.Tensor,
    delta_neg: torch.Tensor,
    sigma_noise: float,
) -> tuple:
    """
    Simulates annotator imprecision by randomly flipping voxels near the
    boundary of edit delta masks.  sigma_noise ∈ [0, 15] voxels.
    Returns noisy (delta_pos, delta_neg) — (B,1,X,Y,Z), same device.
    """
    delta_pos = _to_5d(delta_pos)
    delta_neg = _to_5d(delta_neg)
    if sigma_noise <= 0.0:
        return delta_pos, delta_neg

    def _noise_mask(m: torch.Tensor) -> torch.Tensor:
        """Add Gaussian-weighted boundary noise to a binary mask."""
        m_np = m.squeeze().cpu().numpy()
        if m_np.sum() == 0:
            return m
        # Distance transform → probability of flip decreases with interior distance
        dist = ndimage.distance_transform_edt(m_np).astype(np.float32)
        boundary_prob = np.exp(-(dist ** 2) / (2 * sigma_noise ** 2))
        flip = (np.random.rand(*m_np.shape) < boundary_prob * 0.3).astype(np.float32)
        noisy = np.clip(m_np + flip - 2 * m_np * flip, 0, 1)
        return torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(m.device)

    dp_noisy = torch.cat([_noise_mask(delta_pos[b]) for b in range(delta_pos.size(0))], dim=0)
    dn_noisy = torch.cat([_noise_mask(delta_neg[b]) for b in range(delta_neg.size(0))], dim=0)
    return dp_noisy, dn_noisy


# ─────────────────────────────────────────────────────────────────────────────
# Edit Delta Encoder
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBlock3D(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )


class EditDeltaEncoder(nn.Module):
    """
    Encodes [Δm+, Δm-, y'_{i-1}, x] (4-channel 3D volume) into a
    128-dim latent edit embedding e_i, then appends 4 geometric
    descriptors to produce e_i_full ∈ R^132.
    """
    EMBED_DIM = 128
    FULL_DIM  = 132   # 128 + 4 geometric descriptors

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            _ConvBlock3D(in_channels, 32,  stride=2),
            _ConvBlock3D(32,          64,  stride=2),
            _ConvBlock3D(64,          128, stride=2),
        )
        self.pool    = nn.AdaptiveAvgPool3d(1)
        self.project = nn.Linear(128, self.EMBED_DIM)

    @staticmethod
    def _to_5d(t: torch.Tensor) -> torch.Tensor:
        """Ensure (B,1,X,Y,Z): add channel dim if missing, drop extra channels.

        MONAI MetaTensors returned by the dataloader can be 4-D (B,X,Y,Z)
        when the channel dimension has been squeezed internally.
        """
        t = t.float()
        if t.dim() == 4:          # (B,X,Y,Z) → (B,1,X,Y,Z)
            return t.unsqueeze(1)
        if t.dim() == 5 and t.size(1) != 1:   # multi-channel → take first
            return t[:, :1, ...]
        return t

    def forward(
        self,
        delta_pos: torch.Tensor,   # (B,1,X,Y,Z)
        delta_neg: torch.Tensor,   # (B,1,X,Y,Z)
        prev_pred: torch.Tensor,   # (B,1,X,Y,Z) binary
        image:     torch.Tensor,   # (B,1,X,Y,Z)
        geo_desc:  torch.Tensor,   # (B,4)  — precomputed geometric descriptors
    ) -> torch.Tensor:             # (B, FULL_DIM=132)
        delta_pos = self._to_5d(delta_pos)
        delta_neg = self._to_5d(delta_neg)
        prev_pred = self._to_5d(prev_pred)
        image     = self._to_5d(image)
        x = torch.cat([delta_pos, delta_neg, prev_pred, image], dim=1)  # (B,4,X,Y,Z)
        feat = self.pool(self.encoder(x)).flatten(1)                     # (B,128)
        e    = self.project(feat)                                         # (B,128)
        return torch.cat([e, geo_desc], dim=1)                           # (B,132)


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding (sinusoidal, for iteration index)
# ─────────────────────────────────────────────────────────────────────────────

class _IterationPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_iters: int = 20):
        super().__init__()
        pe = torch.zeros(max_iters, d_model)
        pos = torch.arange(0, max_iters, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)   # (max_iters, d_model)

    def forward(self, x: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        """x: (B, T, d_model)"""
        T = x.size(1)
        return x + self.pe[start_idx: start_idx + T].unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Correction Pattern Memory
# ─────────────────────────────────────────────────────────────────────────────

class CorrectionPatternMemory(nn.Module):
    """
    Maintains a session memory M = [e_1_full, ..., e_{i-1}_full] and
    encodes it via a 2-layer Transformer to produce a memory summary
    vector m_i ∈ R^132 and a persistent-failure flag f_persistent.

    Call reset() at the start of each new case.
    Call update(e_i_full) after each iteration to append and summarise.
    """
    D_MODEL  = EditDeltaEncoder.FULL_DIM   # 132
    PERSIST_THRESHOLD = 0.85

    def __init__(self, max_iters: int = 20):
        super().__init__()
        self.max_iters = max_iters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL,
            nhead=4,
            dim_feedforward=256,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer    = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_enc        = _IterationPositionalEncoding(self.D_MODEL, max_iters)
        self._history: list = []   # list of (B, D_MODEL) tensors

    def reset(self):
        self._history = []

    def update(
        self,
        e_full: torch.Tensor,   # (B, D_MODEL)
    ) -> tuple:                  # (m_i: (B, D_MODEL), f_persistent: bool)
        """
        Appends e_full to the session history, runs the Transformer,
        and returns the summary vector m_i and the persistence flag.
        """
        self._history.append(e_full.detach())   # detach — memory is state, not gradient path

        if len(self._history) == 1:
            # First iteration: no history to summarise yet
            m_i = torch.zeros_like(e_full)
            return m_i, False

        # Stack history: (B, T, D_MODEL)
        history = torch.stack(self._history, dim=1)
        history = self.pos_enc(history)
        summary = self.transformer(history)           # (B, T, D_MODEL)
        m_i     = summary[:, -1, :]                  # take last position as summary

        # Persistent failure: cosine similarity between current and all previous
        f_persistent = False
        if len(self._history) >= 3:
            curr_norm = F.normalize(e_full.detach(), dim=-1)         # (B, D)
            hist_norm = F.normalize(history[:, :-1, :], dim=-1)     # (B, T-1, D)
            sims = (curr_norm.unsqueeze(1) * hist_norm).sum(-1)     # (B, T-1)
            if sims.max().item() > self.PERSIST_THRESHOLD:
                f_persistent = True
                # Store index of most similar past edit for spatial amplification
                self._persistent_idx = int(sims.mean(0).argmax().item())

        return m_i, f_persistent

    def get_persistent_edit(self) -> torch.Tensor | None:
        """Returns the edit embedding most similar to the current one, or None."""
        if hasattr(self, '_persistent_idx') and self._persistent_idx < len(self._history):
            return self._history[self._persistent_idx]
        return None
