import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

from utils.util import _bbox_mask
from utils import scribble, boundary_selection
from models.medsa import (
    EditDeltaEncoder,
    CorrectionPatternMemory,
    compute_geometric_descriptors,
    apply_curriculum_noise,
)
from models.policy import (
    EditConditionedTypePolicy,
    ReplayBuffer,
    compute_dqn_reward,
    ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE, ACTION_STOP,
)
from utils.clinical_priority import priority_mask_from_errors, priority_hit_score
from .trainer_basic import Trainer_basic


# ─────────────────────────────────────────────────────────────────────────────
# Shape normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_5d(t: torch.Tensor) -> torch.Tensor:
    """Guarantee tensor is (B, 1, X, Y, Z).
    Handles:
      (B, X, Y, Z)       → unsqueeze channel dim
      (B, C, X, Y, Z)    → take first channel only (C > 1 edge-case)
    """
    if t.dim() == 4:
        return t.unsqueeze(1)
    if t.dim() == 5 and t.size(1) != 1:
        return t[:, :1, ...]
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Spatial GT heatmap helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_heatmap_3d(
    shape: tuple, centroid: tuple, sigma: float, device
) -> torch.Tensor:
    """3D Gaussian centred at centroid (x, y, z) ∈ [0, shape). Shape (1,1,X,Y,Z)."""
    X, Y, Z = shape
    cx, cy, cz = centroid
    gx = torch.arange(X, dtype=torch.float32, device=device)
    gy = torch.arange(Y, dtype=torch.float32, device=device)
    gz = torch.arange(Z, dtype=torch.float32, device=device)
    gx, gy, gz = torch.meshgrid(gx, gy, gz, indexing='ij')
    dist2 = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2
    return torch.exp(-dist2 / (2 * sigma ** 2)).unsqueeze(0).unsqueeze(0)


def compute_spatial_gt_heatmap(
    delta_pos: torch.Tensor,   # (B,1,X,Y,Z) — FN region of *next* iteration
    delta_neg: torch.Tensor,   # (B,1,X,Y,Z) — FP region of *next* iteration
    sigma: float = 5.0,
    pred: torch.Tensor = None,          # (B,1,X,Y,Z) soft/hard prediction
    image: torch.Tensor = None,         # (B,1,X,Y,Z) intensity volume
    use_clinical_priority: bool = False,
    lam_iso: float = 1.0,
    lam_int: float = 0.5,
    lam_bnd: float = 0.5,
) -> torch.Tensor:             # (B,1,X,Y,Z) ∈ [0,1]
    """
    GT heatmap centred at the centroid of the selected connected component of
    next-iteration edit deltas.  By default uses the largest CC; with
    ``use_clinical_priority`` uses volume × clinical-risk ranking instead.
    """
    from utils.clinical_priority import select_priority_component

    B, _, X, Y, Z = delta_pos.shape
    heatmaps = torch.zeros(B, 1, X, Y, Z, dtype=torch.float32, device=delta_pos.device)

    for b in range(B):
        delta_union = ((delta_pos[b, 0] + delta_neg[b, 0]) > 0).cpu().numpy()
        if not delta_union.any():
            continue

        if use_clinical_priority:
            pred_np = None
            if pred is not None:
                pred_np = (pred[b, 0].detach().float().cpu().numpy() > 0.5)
            else:
                pred_np = np.zeros_like(delta_union, dtype=bool)
            img_np = None
            if image is not None:
                img_np = image[b, 0].detach().float().cpu().numpy()
            _, centroid, _, _ = select_priority_component(
                delta_union, pred_np, img_np,
                lam_iso=lam_iso, lam_int=lam_int, lam_bnd=lam_bnd,
            )
            if centroid is None:
                continue
        else:
            labeled, n_comp = ndimage.label(delta_union)
            if n_comp == 0:
                continue
            comp_sizes  = ndimage.sum(delta_union, labeled, range(1, n_comp + 1))
            largest_idx = int(np.argmax(comp_sizes)) + 1
            centroid    = ndimage.center_of_mass(delta_union, labeled, largest_idx)

        heatmaps[b] = _gaussian_heatmap_3d((X, Y, Z), centroid, sigma, delta_pos.device)

    return heatmaps


def _heatmap_to_click(heatmap: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Sample click coordinates from heatmap (B,1,X,Y,Z) via multinomial sampling.
    Returns (B, k, 3) integer coordinates.
    """
    B, _, X, Y, Z = heatmap.shape
    flat = heatmap.view(B, -1)
    flat = flat / (flat.sum(dim=-1, keepdim=True) + 1e-8)
    indices = torch.multinomial(flat, num_samples=k, replacement=False)   # (B, k)
    x_idx = indices // (Y * Z)
    y_idx = (indices % (Y * Z)) // Z
    z_idx = indices % Z
    return torch.stack([x_idx, y_idx, z_idx], dim=-1)   # (B, k, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Persistent-failure region mask
# ─────────────────────────────────────────────────────────────────────────────

def _build_persist_region(
    persist_embed: torch.Tensor,    # (B, 132)
    fn_mask:       torch.Tensor,    # (B,1,X,Y,Z)
    fp_mask:       torch.Tensor,    # (B,1,X,Y,Z)
) -> torch.Tensor:                  # (B,1,X,Y,Z) binary amplification mask
    """Use the FN∪FP from the most similar past edit as the amplification region."""
    return ((fn_mask + fp_mask) > 0).float()


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum noise schedule
# ─────────────────────────────────────────────────────────────────────────────

def _sigma_noise(epoch: int, sigma_start: float = 15.0, anneal_end: int = 150) -> float:
    """σ_noise(t) = σ_start · max(0, 1 − t / anneal_end)"""
    return sigma_start * max(0.0, 1.0 - epoch / anneal_end)


# ─────────────────────────────────────────────────────────────────────────────
# Loss weight schedules
# ─────────────────────────────────────────────────────────────────────────────

def _lambda_spatial(epoch: int, max_val: float = 0.5, ramp_end: int = 80) -> float:
    """Ramps from 0 → max_val over epochs 1–80."""
    return max_val * min(1.0, epoch / max(ramp_end, 1))


def _lambda_rl(epoch: int, start: int = 50, ramp_len: int = 100) -> float:
    """Ramps from 0 → 1.0 over epochs 51–150."""
    return max(0.0, (epoch - start) / max(ramp_len, 1))


# ─────────────────────────────────────────────────────────────────────────────
# MEDSA Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer(Trainer_basic):

    def __init__(self, args, logger):
        super().__init__(args, logger)

        if getattr(args, 'use_medsa', False):
            # Edit encoder and memory live in the trainer (share device with SAM)
            self.edit_encoder = EditDeltaEncoder(in_channels=4).to(args.device)
            self.memory       = CorrectionPatternMemory(max_iters=args.iter_nums + 2).to(args.device)
            self.type_policy  = EditConditionedTypePolicy(
                state_dim=EditConditionedTypePolicy.STATE_DIM,
            ).to(args.device)
            self.replay_buffer = ReplayBuffer(
                capacity=getattr(args, 'dqn_replay_capacity', 10_000)
            )
            # Policy optimiser (separate from the main SAM optimiser).
            # NOTE: spatial_head is intentionally excluded — it belongs to
            # sam.mask_decoder and is already updated by self.optimizer via
            # the spatial MSE loss.  Including it here would cause two
            # optimizers to fight over the same parameters, destabilising
            # the mask decoder and hurting segmentation quality.
            self.policy_optimizer = torch.optim.AdamW(
                list(self.edit_encoder.parameters()) +
                list(self.memory.parameters()) +
                list(self.type_policy.q_net.parameters()),
                lr=getattr(args, 'medsa_lr', 1e-4),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Main forward pass (called once per training sample / val sample)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        sam_model,
        image,
        label,
        iter_nums,
        train=False,
        return_each_iter=False,
        epoch=0,
    ):
        use_medsa = getattr(self.args, 'use_medsa', False)

        if return_each_iter:
            return_mask_total_iter = torch.zeros(
                [iter_nums, 1, image.size(2), image.size(3), image.size(4)]
            )

        image_embedding, feature_list = self.sam.image_encoder(image)
        self.click_points = []
        self.click_labels = []
        return_loss = 0
        prev_masks  = torch.zeros_like(label, dtype=torch.float).to(label.device)

        # ── Loss accumulators ─────────────────────────────────────────────
        self._train_loss_components = {
            'seg': 0., 'boundary': 0., 'dice_reg': 0.,
            'refine': 0.,
            # MEDSA diagnostics (always present when use_medsa=True)
            'spatial': 0., 'dqn': 0., 'reward': 0.,
            '_epsilon': 0., '_lam_s': 0., '_lam_rl': 0.,
            '_act_pt': 0, '_act_bx': 0, '_act_sc': 0, '_act_st': 0,
            '_episodes': 0,
        }
        actual_iters = 0

        # ── MEDSA session state ───────────────────────────────────────────
        if use_medsa:
            self.memory.reset()
            e_full          = torch.zeros(image.size(0), 132, device=self.args.device)
            m_i             = torch.zeros(image.size(0), 132, device=self.args.device)
            f_persistent    = False
            persist_region  = None
            prev_action        = ACTION_POINT
            prev_state_np      = None
            prev_dice          = 0.0
            prev_geo           = torch.zeros(image.size(0), 4, device=self.args.device)
            prev_spatial_hmap  = None   # cached H_{i-1} for spatial entropy in state
            prev_priority_mask = None   # high-priority CC at previous step (for priority_hit)
            prev_spatial_entropy = 1.0
            sigma_noise = (
                _sigma_noise(epoch,
                    sigma_start=getattr(self.args, 'medsa_noise_sigma_start', 15.0),
                    anneal_end=getattr(self.args,  'medsa_noise_anneal_end',  150))
                if train else 0.0
            )
            lam_s  = _lambda_spatial(epoch,
                        max_val=0.5,
                        ramp_end=getattr(self.args, 'medsa_spatial_ramp_end', 80))
            lam_rl = _lambda_rl(epoch,
                        start=getattr(self.args,    'medsa_rl_start_epoch',   50),
                        ramp_len=getattr(self.args, 'medsa_rl_ramp_len',     100))
            epsilon = max(0.05, 0.5 - epoch * 0.003)   # decaying ε-greedy

            # Snapshot schedule values into diagnostics (overwrite each sample
            # — we want the latest values, not a mean)
            if train:
                self._train_loss_components['_epsilon'] = epsilon
                self._train_loss_components['_lam_s']   = lam_s
                self._train_loss_components['_lam_rl']  = lam_rl

        # ── Interaction loop ──────────────────────────────────────────────
        for iter_num in range(iter_nums):
            actual_iters += 1
            loss = 0
            prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

            # ── Compute edit delta (FN/FP) ────────────────────────────────
            # Normalise to (B,1,X,Y,Z) — MONAI MetaTensor collation can
            # return tensors as (B,X,Y,Z) without an explicit channel dim.
            label_5d              = _ensure_5d(label)
            prev_masks_sigmoid_5d = _ensure_5d(prev_masks_sigmoid)
            image_5d              = _ensure_5d(image)

            pred_bin  = (prev_masks_sigmoid_5d > 0.5)
            true_bin  = (label_5d > 0)
            fn_mask   = (true_bin & ~pred_bin).float()   # Δm+  (B,1,X,Y,Z)
            fp_mask   = (~true_bin & pred_bin).float()   # Δm−  (B,1,X,Y,Z)

            # ── MEDSA: encode edit, query policy ─────────────────────────
            medsa_context = None
            action = ACTION_POINT   # default (non-MEDSA path)
            curr_priority_mask = None
            use_prio = getattr(self.args, 'use_clinical_priority', False)

            if use_medsa:
                # Curriculum noise on edit delta (all tensors (B,1,X,Y,Z))
                if train and sigma_noise > 0:
                    fn_enc, fp_enc = apply_curriculum_noise(
                        fn_mask, fp_mask, sigma_noise
                    )
                else:
                    fn_enc, fp_enc = fn_mask, fp_mask

                # Geometric descriptors → (B,4)
                geo = compute_geometric_descriptors(
                    fn_enc,
                    fp_enc,
                    prev_masks_sigmoid_5d,
                )

                # Encode edit: [Δm+, Δm−, ŷ_{i-1}, image] → (B,4,X,Y,Z) → (B,132)
                e_full = self.edit_encoder(
                    fn_enc,
                    fp_enc,
                    pred_bin.float(),
                    image_5d,
                    geo,
                )   # (B, 132)

                # Update memory
                m_i, f_persistent = self.memory.update(e_full)

                # Persistent region mask
                persist_embed = self.memory.get_persistent_edit()
                if f_persistent and persist_embed is not None:
                    persist_region = _build_persist_region(
                        persist_embed, fn_enc, fp_enc   # already (B,1,X,Y,Z)
                    )
                else:
                    persist_region = None

                # Detach e_full and m_i so that FiLM conditioning in the
                # Spatial Next-Prompt Head uses fixed, inference-mode edit
                # embeddings.  Without detach, policy_optimizer updates to
                # the edit encoder shift the FiLM scale/shift factors applied
                # to the mask decoder's feature maps, destabilising the SAM
                # backbone even though parameter sets are formally separated.
                medsa_context = {
                    'e_full':        e_full.detach(),
                    'm_i':           m_i.detach(),
                    'f_persistent':  f_persistent,
                    'persist_region': persist_region,
                }

                # Clinical priority mask for current FN∪FP (used for prompts + reward).
                # Use clean (unnoised) FN/FP so curriculum speckles do not explode CC count.
                if use_prio:
                    with torch.no_grad():
                        curr_priority_mask, _, _ = priority_mask_from_errors(
                            fn_mask[0], fp_mask[0],
                            prev_masks_sigmoid_5d[0],
                            image=image_5d[0],
                            lam_iso=getattr(self.args, 'prio_lam_iso', 1.0),
                            lam_int=getattr(self.args, 'prio_lam_int', 0.5),
                            lam_bnd=getattr(self.args, 'prio_lam_bnd', 0.5),
                        )

                # Query type policy (only if RL phase active)
                if train and lam_rl > 0:
                    edit_vol  = float(geo[0, 0].item())   # v_i
                    edit_bnd  = float(geo[0, 3].item())   # b_i
                    curr_dice = self.get_dice_score(prev_masks_sigmoid, label)

                    # ── Derive compact 8-scalar state from embeddings ─────────
                    # Scalar 6: normalised error magnitude (stable proxy for
                    # error severity that doesn't shift with encoder updates)
                    e_cnn      = e_full[:, :128].detach()
                    m_cnn      = m_i[:, :128].detach()
                    e_norm     = torch.norm(e_cnn, dim=1).mean().item()
                    e_mag_norm = e_norm / (e_norm + 1.0)

                    # Scalar 7: cosine similarity between current error and
                    # memory → high value = recurring failure pattern
                    persistence = float(
                        F.cosine_similarity(e_cnn, m_cnn, dim=1).mean().item()
                    )

                    # Scalar 8: entropy of previous spatial heatmap H_{i-1}
                    # High entropy → spatial head uncertain → more clicks needed
                    # Low entropy  → spatial head confident → good stop candidate
                    if prev_spatial_hmap is not None:
                        h_flat = torch.sigmoid(prev_spatial_hmap.detach()).flatten().float()
                        h_flat = h_flat / (h_flat.sum() + 1e-8)
                        raw_ent = -(h_flat * torch.log(h_flat + 1e-10)).sum().item()
                        spatial_entropy = raw_ent / math.log(float(h_flat.numel()) + 1.0)
                    else:
                        spatial_entropy = 1.0   # maximum uncertainty at first step

                    state = EditConditionedTypePolicy.build_state(
                        dice_current=float(curr_dice),
                        delta_dice=float(curr_dice) - prev_dice,
                        iter_progress=iter_num / max(iter_nums, 1),
                        edit_volume=edit_vol,
                        edit_bnd_ovlp=edit_bnd,
                        error_magnitude=e_mag_norm,
                        persistence=persistence,
                        spatial_entropy=spatial_entropy,
                        device=self.args.device,
                    )
                    action = self.type_policy.select_action(
                        state, epsilon,
                        edit_volume=edit_vol,
                        iter_idx=iter_num,
                        max_iters=iter_nums,
                        dice_current=float(curr_dice),
                        spatial_entropy=spatial_entropy,
                        persistence=persistence,
                        use_uncertainty_gate=getattr(self.args, 'use_uncertainty_gate', False),
                        uncert_low=getattr(self.args, 'uncert_low', 0.35),
                        uncert_high=getattr(self.args, 'uncert_high', 0.65),
                        persist_thresh=getattr(self.args, 'persist_thresh', 0.85),
                    )

                    # Count action type for logging
                    _act_key = {
                        ACTION_POINT:   '_act_pt',
                        ACTION_BOX:     '_act_bx',
                        ACTION_SCRIBBLE:'_act_sc',
                        ACTION_STOP:    '_act_st',
                    }.get(action, '_act_pt')
                    self._train_loss_components[_act_key] = (
                        self._train_loss_components.get(_act_key, 0) + 1
                    )

                    # Store previous state for replay (we record reward at next step)
                    if prev_state_np is not None:
                        # Residual spatial grounding of H_{i-1} vs current FN∪FP
                        spatial_hit = 0.0
                        corr_mask = (fn_enc + fp_enc).clamp(0.0, 1.0)
                        if prev_spatial_hmap is not None:
                            with torch.no_grad():
                                corr_size = corr_mask.sum().item() + 1e-8
                                spatial_hit = float(
                                    (torch.sigmoid(prev_spatial_hmap) * corr_mask
                                     ).sum().item() / corr_size
                                )

                        # Priority hit: fraction of current correction on the
                        # *previous* step's high-priority CC (what we aimed at)
                        prio_hit = 0.0
                        if use_prio and prev_priority_mask is not None:
                            prio_hit = priority_hit_score(corr_mask[0], prev_priority_mask)

                        reward = compute_dqn_reward(
                            prev_action,
                            dice_current=float(curr_dice),
                            dice_prev=prev_dice,
                            edit_volume=float(prev_geo[0, 0].item()),
                            edit_bnd_ovlp=float(prev_geo[0, 3].item()),
                            spatial_hit=spatial_hit,
                            priority_hit=prio_hit,
                            spatial_entropy=prev_spatial_entropy,
                            use_clinical_priority=use_prio,
                            use_uncertainty_gate=getattr(self.args, 'use_uncertainty_gate', False),
                            beta_priority=getattr(self.args, 'beta_priority', 0.30),
                            gamma_uncert=getattr(self.args, 'gamma_uncert', 0.20),
                        )
                        # Accumulate reward for logging
                        self._train_loss_components['reward'] = (
                            self._train_loss_components.get('reward', 0.) + reward
                        )
                        self.replay_buffer.push(
                            prev_state_np,
                            prev_action,
                            reward,
                            state.detach().cpu().numpy().squeeze(0),
                            done=False,
                        )

                    prev_state_np = state.detach().cpu().numpy().squeeze(0)
                    prev_action   = action
                    prev_dice     = float(curr_dice)
                    prev_geo      = geo.detach()
                    prev_spatial_entropy = spatial_entropy
                    if curr_priority_mask is not None:
                        prev_priority_mask = curr_priority_mask.detach()

                # Stop action → finish loop
                if action == ACTION_STOP and iter_num > 0:
                    actual_iters -= 1   # don't count the stop step
                    break

            # ── Get physician prompts (action-conditioned) ────────────────
            points_input, labels_input, box_input = self.get_points(
                prev_masks_sigmoid, label,
                train_mode=train,
                action=action,
                heatmap=medsa_context['e_full'] if (
                    use_medsa and medsa_context is not None and action == ACTION_POINT
                ) else None,
                prev_fn=fn_enc if use_medsa else None,
                prev_fp=fp_enc if use_medsa else None,
                priority_mask=curr_priority_mask if use_medsa else None,
                image_vol=image_5d if use_medsa else None,
            )

            # ── Model forward ─────────────────────────────────────────────
            mask, dice_pred, spatial_hmap, iou_token = \
                self.iteration_forward(
                    sam_model, feature_list, image_embedding, prev_masks,
                    points=[points_input, labels_input], boxes=box_input,
                    medsa_context=medsa_context,
                )

            # Cache H_i for use as spatial_entropy scalar at next iteration
            if use_medsa and spatial_hmap is not None:
                prev_spatial_hmap = spatial_hmap.detach()

            # ── Select best candidate mask (by IoU score) ────────────────
            B = mask.size(0)
            if self.args.multiple_outputs:
                _, max_label_index = torch.max(dice_pred, dim=1)
                b_idx              = torch.arange(B, device=mask.device)
                mask_best          = mask[b_idx, max_label_index].unsqueeze(1)
            else:
                mask_best = mask

            # ── Training path ─────────────────────────────────────────────
            if train:
                iter_components = {'seg': 0., 'boundary': 0., 'dice_reg': 0., 'spatial': 0.}
                n_cands = 0

                if self.args.multiple_outputs:
                    for i in range(mask.size(1)):
                        l, ld = self.calculate_loss(
                            mask[:, i, :].unsqueeze(1), prev_masks,
                            dice_pred[:, i], label, labels_input, iter_num,
                        )
                        loss += l
                        for k in ld:
                            iter_components[k] = iter_components.get(k, 0.) + ld[k]
                        n_cands += 1
                    if n_cands > 0:
                        for k in iter_components:
                            iter_components[k] /= n_cands
                else:
                    loss, ld = self.calculate_loss(
                        mask, prev_masks, dice_pred[:, 0], label, labels_input, iter_num,
                    )
                    iter_components.update(ld)

                # ── MEDSA spatial loss ─────────────────────────────────────
                if use_medsa and spatial_hmap is not None and lam_s > 0:
                    # GT heatmap = Gaussian at centroid of next FN∪FP
                    with torch.no_grad():
                        next_pred_bin  = (torch.sigmoid(mask_best) > 0.5)
                        next_fn        = (true_bin & ~next_pred_bin).float()
                        next_fp        = (~true_bin & next_pred_bin).float()
                    gt_hmap = compute_spatial_gt_heatmap(
                        next_fn.unsqueeze(1) if next_fn.dim() == 4 else next_fn,
                        next_fp.unsqueeze(1) if next_fp.dim() == 4 else next_fp,
                        sigma=getattr(self.args, 'spatial_sigma', 5.0),
                        pred=prev_masks_sigmoid_5d,
                        image=image_5d,
                        use_clinical_priority=getattr(self.args, 'use_clinical_priority', False),
                        lam_iso=getattr(self.args, 'prio_lam_iso', 1.0),
                        lam_int=getattr(self.args, 'prio_lam_int', 0.5),
                        lam_bnd=getattr(self.args, 'prio_lam_bnd', 0.5),
                    )
                    l_spatial = self.loss_spatial(spatial_hmap, gt_hmap) * lam_s
                    loss     += l_spatial
                    iter_components['spatial'] = float(l_spatial.item())

                # Accumulate
                for k in iter_components:
                    self._train_loss_components[k] = (
                        self._train_loss_components.get(k, 0.) + iter_components[k]
                    )
                if self.args.refine:
                    if self.args.no_detach:
                        mask_refine, _ = self.sam.mask_decoder.refine(
                            image, mask_best, [self.click_points, self.click_labels], mask_best)
                    else:
                        mask_refine, _ = self.sam.mask_decoder.refine(
                            image, mask_best, [self.click_points, self.click_labels],
                            mask_best.detach())
                    refine_l  = self.loss_segmentation(mask_refine, label)
                    loss     += refine_l
                    self._train_loss_components['refine'] += refine_l.item()
                    mask_best  = mask_refine

            # ── Validation / inference path ───────────────────────────────
            else:
                if self.args.refine:
                    if self.args.no_detach:
                        mask_refine, _ = self.sam.mask_decoder.refine(
                            image, mask_best, [self.click_points, self.click_labels], mask_best)
                    else:
                        mask_refine, _ = self.sam.mask_decoder.refine(
                            image, mask_best, [self.click_points, self.click_labels],
                            mask_best.detach())
                    mask_best = mask_refine

                loss = self.get_dice_score(torch.sigmoid(mask_best), label)


            return_loss += loss
            prev_masks   = mask_best

            if return_each_iter:
                return_mask_total_iter[iter_num, :] = mask_best

        # ── Terminal DQN experience (stop / end of episode) ───────────────
        if use_medsa and train and lam_rl > 0 and prev_state_np is not None:
            final_dice = self.get_dice_score(torch.sigmoid(mask_best), label)
            reward     = compute_dqn_reward(
                prev_action,
                dice_current=float(final_dice),
                dice_prev=prev_dice,
                edit_volume=float(prev_geo[0, 0].item()),
                edit_bnd_ovlp=float(prev_geo[0, 3].item()),
                spatial_hit=0.0,
                priority_hit=0.0,
                spatial_entropy=prev_spatial_entropy,
                use_clinical_priority=getattr(self.args, 'use_clinical_priority', False),
                use_uncertainty_gate=getattr(self.args, 'use_uncertainty_gate', False),
                beta_priority=getattr(self.args, 'beta_priority', 0.30),
                gamma_uncert=getattr(self.args, 'gamma_uncert', 0.20),
            )
            dummy_next = np.zeros_like(prev_state_np)
            self.replay_buffer.push(prev_state_np, prev_action, reward, dummy_next, done=True)
            # Accumulate terminal reward and episode count
            self._train_loss_components['reward'] = (
                self._train_loss_components.get('reward', 0.) + reward
            )
            self._train_loss_components['_episodes'] = (
                self._train_loss_components.get('_episodes', 0) + 1
            )

            # Store lam_rl so trainer_basic.train() can run the DQN update
            # OUTSIDE amp.autocast() — calling loss.backward() inside autocast
            # causes PyTorch to lose grad_fn on the Q-network output.
            self._last_lam_rl = lam_rl
            # dqn loss component will be filled in by trainer_basic.train()
            self._train_loss_components['dqn'] = 0.0

        # Normalise per-iteration averages (exclude counters, snapshots, and reward)
        # reward is already per-episode (not per-step), normalised below by _episodes
        _no_avg = {'dqn', '_epsilon', '_lam_s', '_lam_rl',
                   '_act_pt', '_act_bx', '_act_sc', '_act_st', '_episodes',
                   'reward'}   # reward normalised per-episode, not per-step
        if train and actual_iters > 0:
            for k in self._train_loss_components:
                if k not in _no_avg:
                    self._train_loss_components[k] /= actual_iters

        # Reward: normalise by episode count (1 per forward() call when RL active)
        eps_count = self._train_loss_components.get('_episodes', 0)
        if eps_count > 0:
            self._train_loss_components['reward'] /= eps_count

        # Store iteration count for CORA monitoring
        self._clinical_iters_used = actual_iters

        if return_each_iter:
            return return_loss / max(actual_iters, 1), return_mask_total_iter
        return return_loss / max(actual_iters, 1), prev_masks

    # ─────────────────────────────────────────────────────────────────────────
    # iteration_forward (wraps the SAM model forward pass)
    # ─────────────────────────────────────────────────────────────────────────

    def iteration_forward(
        self, sam_model, features, image_embedding, prev_masks,
        points=None, boxes=None, medsa_context=None,
    ):
        prev_masks = F.interpolate(prev_masks, scale_factor=0.25)
        features   = [features[i].to(self.args.device) for i in range(len(features))]

        new_point_embedding, new_image_embedding = sam_model.prompt_encoder(
            points=points, boxes=boxes,
            masks=prev_masks,
            image_embeddings=image_embedding.to(self.args.device),
        )
        mask, dice_pred, spatial_hmap, iou_token = \
            sam_model.mask_decoder(
                prompt_embeddings=new_point_embedding,
                image_embeddings=new_image_embedding,
                feature_list=features,
                medsa_context=medsa_context,
            )
        return mask, dice_pred, spatial_hmap, iou_token

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt sampling (action-conditioned)
    # ─────────────────────────────────────────────────────────────────────────

    def get_points(
        self,
        prev_masks,
        label,
        train_mode=True,
        action=ACTION_POINT,
        heatmap=None,
        prev_fn=None,
        prev_fp=None,
        priority_mask=None,
        image_vol=None,
    ):
        mode = 'train' if train_mode else 'validation'
        use_priority_prompts = getattr(self.args, 'use_medsa', False)

        if action == ACTION_BOX or not use_priority_prompts:
            # Always provide box when action=BOX (or agent disabled → standard path)
            bbox_coords = _bbox_mask(
                label[:, 0, :], mode=mode,
                dynamic=getattr(self.args, 'dynamic_box', False),
            ).to(self.args.device) if self.args.use_box or action == ACTION_BOX else None
            batch_points, batch_labels = self.get_next_point(prev_masks, label, mode=mode)
        else:
            bbox_coords  = None
            batch_points, batch_labels = self._get_next_point_medsa(
                prev_masks, label, mode=mode, action=action,
                prev_fn=prev_fn, prev_fp=prev_fp,
                priority_mask=priority_mask,
            )

        points_co = torch.cat(batch_points, dim=0).to(self.args.device)
        points_la = torch.cat(batch_labels, dim=0).to(self.args.device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        if bbox_coords is None and self.args.use_box and action != ACTION_SCRIBBLE:
            bbox_coords = _bbox_mask(
                label[:, 0, :], mode=mode,
                dynamic=getattr(self.args, 'dynamic_box', False),
            ).to(self.args.device)

        return points_co, points_la, bbox_coords

    def _get_next_point_medsa(self, prev_seg, label, mode, action, prev_fn, prev_fp,
                              priority_mask=None):
        """
        Action-conditioned sampler:
          ACTION_POINT   → pick from FN∪FP (or clinical-priority CC when enabled)
          ACTION_SCRIBBLE → follow skeleton of top-10% FN∪FP (restricted to priority CC)
          ACTION_STOP     → should never reach here (caught in forward loop)
        """
        batch_points, batch_labels = [], []
        pred_masks = (prev_seg > 0.5)
        true_masks = (label  > 0)
        fn_masks   = torch.logical_and(true_masks,  torch.logical_not(pred_masks))
        fp_masks   = torch.logical_and(torch.logical_not(true_masks), pred_masks)

        num_clicks = self.args.num_clicks if mode == 'train' else self.args.num_clicks_validation
        use_prio = (
            getattr(self.args, 'use_clinical_priority', False)
            and priority_mask is not None
            and priority_mask.sum() > 0
        )

        for i in range(label.shape[0]):
            bp_list, bl_list = [], []

            # Restrict sampling to high-priority CC when clinical priority is on
            if use_prio:
                prio = priority_mask
                while prio.dim() < fn_masks[i].dim():
                    prio = prio.unsqueeze(0)
                prio = (prio > 0.5).to(fn_masks[i].device)
                fn_i = fn_masks[i] & prio
                fp_i = fp_masks[i] & prio
                # Fallback if priority CC empty after mask
                if (fn_i | fp_i).sum() == 0:
                    fn_i, fp_i = fn_masks[i], fp_masks[i]
            else:
                fn_i, fp_i = fn_masks[i], fp_masks[i]

            if action == ACTION_SCRIBBLE and getattr(self.args, 'use_scribble', False):
                # Scribble along top-10% of (priority-restricted) FN∪FP voxels
                error_vol = (fn_i.float() + fp_i.float()).clamp(0, 1)
                k_top = max(1, int(error_vol.numel() * 0.10))
                threshold = error_vol.flatten().topk(k=k_top).values.min()

                from utils import scribble as scribble_util
                scribble_obj = scribble_util.CenterlineScribble()
                for raw_region, polarity in [
                    (fn_i.float() * (error_vol >= threshold).float(), 1),
                    (fp_i.float() * (error_vol >= threshold).float(), 0),
                ]:
                    if raw_region.sum() == 0:
                        continue
                    region_input = raw_region.permute(3, 0, 1, 2)   # (Z,1,X,Y)
                    sm = scribble_obj.batch_scribble(region_input).permute(1, 2, 3, 0) > 0
                    coords = torch.argwhere(sm)[:, 1:].unsqueeze(0)
                    if coords.numel() > 0:
                        limit = min(coords.size(1), 500)
                        coords = coords[:, :limit, :]
                        bl_list.append(torch.full((1, limit), float(polarity)))
                        bp_list.append(coords)

            # Fallback / primary: point clicks from FN∪FP (priority-restricted)
            if not bp_list:
                to_point_mask = torch.logical_or(fn_i, fp_i)
                points = torch.argwhere(to_point_mask)
                if len(points) == 0:
                    points = torch.argwhere(true_masks[i])
                if len(points) == 0:
                    points = label.new_zeros(1, 4, dtype=torch.long)

                click_size  = max(min(num_clicks, len(points)), 1)
                click_size  = random.randint(1, click_size) if (
                    self.args.dynamic and mode == 'train'
                ) else click_size
                selected    = points[np.random.choice(len(points), click_size, replace=False)]

                for ci in range(len(selected)):
                    pt = selected[ci]
                    is_pos = bool(fn_masks[i, 0, pt[1], pt[2], pt[3]])
                    bp_list.append(pt[1:].clone().detach().reshape(1, 1, 3))
                    bl_list.append(torch.tensor([float(int(is_pos))]).reshape(1, 1))

            batch_points.append(torch.cat(bp_list, dim=1))
            batch_labels.append(torch.cat(bl_list, dim=1))

        # Pad to same length
        smallest_n = min(t.size(1) for t in batch_labels)
        batch_points = [t[:, :smallest_n] for t in batch_points]
        batch_labels = [t[:, :smallest_n] for t in batch_labels]
        return batch_points, batch_labels

    # ─────────────────────────────────────────────────────────────────────────
    # Base-class get_next_point (unchanged from original src)
    # ─────────────────────────────────────────────────────────────────────────

    def get_next_point(self, prev_seg, label, mode='train'):
        batch_points = []
        batch_labels = []

        pred_masks = (prev_seg > 0.5)
        true_masks = (label > 0)
        fn_masks   = torch.logical_and(true_masks, torch.logical_not(pred_masks))
        fp_masks   = torch.logical_and(torch.logical_not(true_masks), pred_masks)
        to_point_mask = torch.logical_or(fn_masks, fp_masks)

        sample_method = 'center'
        scribble_types = {
            'line':    'LineScribble',
            'center':  'CenterlineScribble',
            'default': 'ContourScribble',
        }

        def create_scribble_mask(scribble_type, data):
            from utils import scribble as sc
            obj  = getattr(sc, scribble_type)()
            mask = obj.batch_scribble(data).permute(1, 2, 3, 0)
            return mask > 0

        points_list = [len(torch.argwhere(to_point_mask[i])) for i in range(to_point_mask.size(0))]
        points_min  = min(points_list)
        num_clicks  = self.args.num_clicks if mode == 'train' else self.args.num_clicks_validation
        click_size  = max(points_min if num_clicks > points_min else num_clicks, 1)
        dynamic_size = random.randint(1, click_size) if self.args.dynamic and mode == 'train' else click_size

        for i in range(label.shape[0]):
            bp_list, bl_list = [], []
            points = torch.argwhere(to_point_mask[i])
            if len(points) == 0:
                points = torch.argwhere(true_masks[i])
            if len(points) == 0:
                points = label.new_zeros(1, 4, dtype=torch.long)

            actual_size  = min(dynamic_size, len(points))
            point_index  = np.random.choice(len(points), size=actual_size, replace=False)
            points_select = points[point_index]

            for click_index in range(actual_size):
                point     = points_select[click_index]
                is_positive = bool(fn_masks[i, 0, point[1], point[2], point[3]])
                bp_list.append(point[1:].clone().detach().reshape(1, 1, 3))
                bl_list.append(torch.tensor([float(int(is_positive))]).reshape(1, 1))

            if self.args.use_scribble:
                from utils.util import _bbox_mask as _bm
                fg       = fn_masks[i].permute(3, 0, 1, 2).float()
                bg_orig  = fp_masks[i].permute(3, 0, 1, 2).float()
                if label[i, 0, :].sum() == 0:
                    batch_points.append(torch.cat(bp_list, dim=1))
                    batch_labels.append(torch.cat(bl_list, dim=1))
                    continue
                bbx   = _bm(label[i, 0, :].unsqueeze(0))
                diff_ = 15
                i_min, i_max = max(0, bbx[:, :, 0] - diff_), min(bbx[:, :, 3] + diff_, 126)
                j_min, j_max = max(0, bbx[:, :, 1] - diff_), min(bbx[:, :, 4] + diff_, 126)
                k_min, k_max = max(0, bbx[:, :, 2] - diff_), min(bbx[:, :, 5] + diff_, 126)
                bg_mask = torch.zeros_like(bg_orig).permute(1, 2, 3, 0)
                bg_mask[:, i_min:i_max, j_min:j_max, k_min:k_max] = 1
                bg = bg_orig * bg_mask.permute(3, 0, 1, 2)
                st = scribble_types.get(sample_method, scribble_types['default'])
                sm_fg = create_scribble_mask(st, fg)
                limit = 500
                fg_coors = torch.argwhere(sm_fg)[:, 1:].unsqueeze(0)
                if fg_coors.size(1) > limit + 50:
                    rn = random.randint(0, fg_coors.size(1) - limit)
                    fg_coors = fg_coors[:, rn: rn + limit, :]
                bp_list.append(fg_coors)
                bl_list.append(torch.ones(1, fg_coors.size(1)))
                sm_bg = create_scribble_mask(st, bg)
                bg_coors = torch.argwhere(sm_bg)[:, 1:].unsqueeze(0)
                if bg_coors.size(1) > limit + 50:
                    rn = random.randint(0, bg_coors.size(1) - limit)
                    bg_coors = bg_coors[:, rn: rn + limit, :]
                bp_list.append(bg_coors)
                bl_list.append(torch.zeros(1, bg_coors.size(1)))

            batch_points.append(torch.cat(bp_list, dim=1))
            batch_labels.append(torch.cat(bl_list, dim=1))

        if self.args.use_scribble:
            smallest_n   = min(t.size(1) for t in batch_labels)
            batch_points = [t[:, :smallest_n] for t in batch_points]
            batch_labels = [t[:, :smallest_n] for t in batch_labels]
        return batch_points, batch_labels

