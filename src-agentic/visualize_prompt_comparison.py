"""
visualize_prompt_comparison.py

For a given test case, runs the CoCC agent step-by-step.  At every
interaction step the backbone is *also* run with the two alternative
prompt types (from the identical previous-segmentation state), so we can
compare what each action would have produced at that moment.

The script then:
  1. Auto-selects the step where the agent's choice most outperforms
     the alternatives (or --viz_step N can fix it manually).
  2. Saves a publication-quality figure showing image + GT and the three
     action outcomes side-by-side with their Dice scores.

Usage
-----
  python visualize_prompt_comparison.py \\
      --data colon \\
      --checkpoint /path/to/best.pth.tar \\
      --case_idx 3 \\
      --save_dir ./viz_output \\
      --viz_step 2          # optional; omit to auto-select the best step
      --n_cases 1           # how many cases to visualise in one run

All other flags (--use_medsa, --use_box, --use_scribble, --iter_nums …)
are inherited from the standard config_args system.
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config.config_args import *   # imports the module-level `parser` object
from config.config_setup import build_model, get_dataloader
from utils.util import _bbox_mask
from utils import scribble
from models.medsa import EditDeltaEncoder, CorrectionPatternMemory
from models.policy import (
    EditConditionedTypePolicy,
    ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE, ACTION_STOP,
)
from processor.trainer import compute_geometric_descriptors, _ensure_5d

# ── visual style ────────────────────────────────────────────────────────────
ACTION_NAME  = {ACTION_POINT: 'Point', ACTION_BOX: 'Box', ACTION_SCRIBBLE: 'Scribble'}
ACTION_COLOR = {ACTION_POINT: '#2196F3', ACTION_BOX: '#FF9800', ACTION_SCRIBBLE: '#4CAF50'}
GT_COLOR     = 'white'
PRED_ALPHA   = 0.35


# ── helpers ─────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_coef(pred_sig: torch.Tensor, label: torch.Tensor) -> float:
    pred_bin = (pred_sig > 0.5).float()
    tp = (pred_bin * label).sum()
    denom = pred_bin.sum() + label.sum()
    return (2 * tp / (denom + 1e-6)).item()


def best_slice(volume_3d: torch.Tensor) -> int:
    """Axial slice index with the most foreground voxels."""
    counts = volume_3d.sum(dim=(1, 2))   # [D]
    return int(counts.argmax().item())


def normalise_slice(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


# ── single backbone forward with a given action ──────────────────────────────

def forward_one_action(
    sam_model, image_embedding, feature_list, prev_masks, label, action, args
):
    """
    Re-run one SAM forward pass with a freshly generated prompt for `action`.
    prev_masks and label must be on GPU (args.device).
    Returns (mask_sigmoid_cpu [1,1,D,H,W], dice_float)
    """
    dev = args.device
    prev_sig = torch.sigmoid(prev_masks)
    pred_bin = (prev_sig > 0.5)
    true_bin = (label > 0)
    fn_masks = (true_bin & ~pred_bin)
    fp_masks = (~true_bin & pred_bin)

    bp_list, bl_list = [], []

    # ── scribble ───────────────────────────────────────────────────────────
    if action == ACTION_SCRIBBLE and getattr(args, 'use_scribble', False):
        error_vol = (fn_masks[0].float() + fp_masks[0].float()).clamp(0, 1)
        k_top     = max(1, int(error_vol.numel() * 0.10))
        threshold = error_vol.flatten().topk(k=k_top).values.min()
        scribble_obj = scribble.CenterlineScribble()
        for region, polarity in [
            (fn_masks[0].float() * (error_vol >= threshold).float(), 1),
            (fp_masks[0].float() * (error_vol >= threshold).float(), 0),
        ]:
            if region.sum() == 0:
                continue
            sm = scribble_obj.batch_scribble(region.permute(3, 0, 1, 2)).permute(1, 2, 3, 0) > 0
            coords = torch.argwhere(sm)[:, 1:].unsqueeze(0)
            if coords.numel() > 0:
                coords = coords[:, :min(coords.size(1), 500), :]
                bl_list.append(torch.full((1, coords.size(1)), float(polarity)))
                bp_list.append(coords)

    # ── point fallback (also used for ACTION_POINT and ACTION_BOX secondary cue) ──
    if not bp_list:
        to_pt  = (fn_masks[0] | fp_masks[0])
        pts    = torch.argwhere(to_pt)
        if len(pts) == 0:
            pts = torch.argwhere(true_bin[0])
        if len(pts) == 0:
            pts = label.new_zeros(1, 4, dtype=torch.long)
        n_clicks = max(min(getattr(args, 'num_clicks_validation', 1), len(pts)), 1)
        sel      = pts[np.random.choice(len(pts), n_clicks, replace=False)]
        for ci in range(len(sel)):
            pt     = sel[ci]
            is_pos = bool(fn_masks[0, 0, pt[1], pt[2], pt[3]])
            bp_list.append(pt[1:].clone().detach().reshape(1, 1, 3))
            bl_list.append(torch.tensor([float(int(is_pos))]).reshape(1, 1))

    points_co = torch.cat(bp_list, dim=1).to(dev)
    labels_co = torch.cat(bl_list, dim=1).to(dev)
    # Match test.py get_points: box cue for BOX and POINT actions (not SCRIBBLE)
    bbox = _bbox_mask(label[:, 0, :]).to(dev) if (args.use_box and action != ACTION_SCRIBBLE) else None

    prev_down   = F.interpolate(prev_masks, scale_factor=0.25)
    feat_dev    = [f.to(dev) for f in feature_list]
    pt_emb, img_emb = sam_model.prompt_encoder(
        points=[points_co, labels_co], boxes=bbox,
        masks=prev_down, image_embeddings=image_embedding.to(dev),
    )
    mask, pred_dice_sc, _, _ = sam_model.mask_decoder(
        prompt_embeddings=pt_emb, image_embeddings=img_emb, feature_list=feat_dev,
    )
    if getattr(args, 'multiple_outputs', False):
        _, idx = torch.max(pred_dice_sc, dim=1)
        mask   = mask[torch.arange(mask.size(0)), idx].unsqueeze(1)

    mask_sig = torch.sigmoid(mask).detach().cpu()
    dice     = dice_coef(mask_sig, label.cpu())
    return mask_sig, dice


# ── independent full run with a fixed action type ────────────────────────────

def run_fixed_action(sam, image, label, fixed_action, n_iters, args):
    """
    Run the backbone for n_iters steps using ONLY fixed_action as the prompt type.
    Each step feeds the previous step's mask as prev_masks (fully sequential, no policy).
    Returns (final_mask_sig_cpu [1,1,D,H,W], final_dice, per_step_dice_list)
    """
    dev = args.device
    image_embedding, feature_list = sam.image_encoder(image)
    prev_masks = torch.zeros_like(label).to(dev)
    per_step_dice = []

    for it in range(n_iters):
        prev_sig = torch.sigmoid(prev_masks) if it > 0 else prev_masks
        mask_sig, dice = forward_one_action(
            sam, image_embedding, feature_list, prev_masks, label, fixed_action, args
        )
        per_step_dice.append(dice)
        # Advance: convert sigmoid mask back to logit space
        prev_masks = torch.logit(
            mask_sig.to(dev).clamp(1e-6, 1 - 1e-6)
        )

    return mask_sig, per_step_dice[-1], per_step_dice


# ── agent interaction with per-step counterfactuals ──────────────────────────

def interaction_with_counterfactuals(sam, edit_encoder, memory, type_policy, image, label, args):
    """
    Run the full CoCC agent interaction loop.
    At every step (i ≥ 1, before the policy could say STOP) we also run the
    backbone with the OTHER two action types from the same prev_masks state.

    Returns list of step-records:
        [{
            'step':         step index (1-based),
            'agent_action': action chosen by policy,
            'results':      {action: (mask_sig_cpu, dice)},  # all 3 actions
            'prev_masks':   prev_masks cpu tensor before this step,
        }]
    """
    dev       = args.device
    max_iters = args.iter_nums
    image_embedding, feature_list = sam.image_encoder(image)

    prev_masks = torch.zeros_like(label).to(dev)
    memory.reset()
    e_full            = torch.zeros(image.size(0), 132, device=dev)
    m_i               = torch.zeros(image.size(0), 132, device=dev)
    prev_dice         = 0.0
    prev_spatial_hmap = None
    step_records      = []

    for iter_num in range(max_iters):
        prev_sig = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

        # ── policy decision ────────────────────────────────────────────────
        action = ACTION_POINT
        if iter_num > 0:
            label_5d = _ensure_5d(label)
            pred_5d  = _ensure_5d(prev_sig)
            image_5d = _ensure_5d(image)
            pred_bin = (pred_5d > 0.5)
            true_bin = (label_5d > 0)
            fn_mask  = (true_bin & ~pred_bin).float()
            fp_mask  = (~true_bin & pred_bin).float()

            with torch.no_grad():
                geo    = compute_geometric_descriptors(fn_mask, fp_mask, pred_5d)
                e_full = edit_encoder(fn_mask, fp_mask, pred_bin.float(), image_5d, geo)
                m_i, _ = memory.update(e_full)
                e_cnn      = e_full[:, :128]
                m_cnn      = m_i[:, :128]
                e_norm     = torch.norm(e_cnn, dim=1).mean().item()
                e_mag      = e_norm / (e_norm + 1.0)
                persistence= float(F.cosine_similarity(e_cnn, m_cnn, dim=1).mean().item())

                if prev_spatial_hmap is not None:
                    h_flat = torch.sigmoid(prev_spatial_hmap).flatten().float()
                    h_flat = h_flat / (h_flat.sum() + 1e-8)
                    raw_ent = -(h_flat * torch.log(h_flat + 1e-10)).sum().item()
                    spatial_entropy = raw_ent / math.log(float(h_flat.numel()) + 1.0)
                else:
                    spatial_entropy = 1.0

                curr_dice = dice_coef(prev_sig.cpu(), label.cpu())
                state = EditConditionedTypePolicy.build_state(
                    dice_current=curr_dice,
                    delta_dice=curr_dice - prev_dice,
                    iter_progress=iter_num / max(max_iters, 1),
                    edit_volume=float(geo[0, 0].item()),
                    edit_bnd_ovlp=float(geo[0, 3].item()),
                    error_magnitude=e_mag,
                    persistence=persistence,
                    spatial_entropy=spatial_entropy,
                    device=dev,
                )
                action = type_policy.select_action(
                    state, epsilon=0.0,
                    edit_volume=float(geo[0, 0].item()),
                    iter_idx=iter_num,
                    max_iters=max_iters,
                    dice_current=curr_dice,
                )
            prev_dice = curr_dice

        if action == ACTION_STOP:
            print(f'  → Agent stopped at iter {iter_num}')
            break

        # ── counterfactual forward passes for all three prompt types ───────
        record = {
            'step':         iter_num + 1,
            'agent_action': action,
            'results':      {},
            'prev_masks':   prev_masks.cpu().clone(),
        }
        for act in [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]:
            with torch.no_grad():
                mask_sig, dice = forward_one_action(
                    sam, image_embedding, feature_list, prev_masks, label, act, args
                )
            record['results'][act] = (mask_sig, dice)
        step_records.append(record)

        # ── advance with the agent's actual choice ─────────────────────────
        prev_masks = record['results'][action][0].to(dev).clone()
        # Undo sigmoid to keep logit-space as SAM expects
        prev_masks = torch.logit(prev_masks.clamp(1e-6, 1 - 1e-6))

        print(
            f'  step {iter_num+1:2d} | agent: {ACTION_NAME[action]:8s} | '
            + ' | '.join(
                f'{ACTION_NAME[a]}: {record["results"][a][1]:.3f}'
                for a in [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]
            )
        )

    return step_records


# ── figure ───────────────────────────────────────────────────────────────────

def make_final_figure(image, label, results_by_action, agent_action, n_iters_agent, case_idx, save_path,
                      column_order=None):
    """
    Show three fully-independent runs (point / box / scribble), each from zero,
    run for their respective number of steps. Agent's action is highlighted.

    results_by_action: {action: (mask_sig_cpu, final_dice, per_step_dice)}
    """
    img_vol = image[0, 0].cpu() if image.ndim == 5 else image[0].cpu()
    lbl_vol = label[0, 0].cpu() if label.ndim == 5 else label[0].cpu()

    sl      = best_slice(lbl_vol)
    img_np  = normalise_slice(img_vol[sl].numpy())
    lbl_np  = lbl_vol[sl].numpy()

    actions = column_order if column_order is not None else [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]
    n_iters = {a: len(results_by_action[a][2]) for a in actions}

    n_cols = 1 + len(actions)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.8 * n_cols, 4.5), facecolor='#1a1a2e')
    fig.suptitle(
        f'Case {case_idx}  ·  Independent fixed-action runs  ·  Agent chose: {ACTION_NAME[agent_action]}',
        color='white', fontsize=11, fontweight='bold', y=1.01
    )

    def style_ax(ax, title, title_color='white', highlight=False):
        ax.set_title(title, color=title_color, fontsize=10,
                     fontweight='bold' if highlight else 'normal', pad=6)
        ax.axis('off')
        if highlight:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(ACTION_COLOR[agent_action])
                spine.set_linewidth(5)

    # column 0: GT
    ax = axes[0]
    ax.imshow(img_np, cmap='gray', interpolation='bilinear')
    ax.contour(lbl_np, levels=[0.5], colors=[GT_COLOR], linewidths=1.5)
    style_ax(ax, 'Image + GT', 'white')

    for col, act in enumerate(actions, start=1):
        mask_sig, final_dice, step_dices = results_by_action[act]
        if mask_sig.ndim == 5:
            pred_np = mask_sig[0, 0, sl].numpy()
        else:
            pred_np = mask_sig[0, sl].numpy()

        is_agent = (act == agent_action)
        ax = axes[col]
        ax.imshow(img_np, cmap='gray', interpolation='bilinear')
        ax.contour(lbl_np, levels=[0.5], colors=[GT_COLOR], linewidths=1.2, linestyles='--')

        overlay = np.zeros((*pred_np.shape, 4))
        r, g, b = tuple(int(ACTION_COLOR[act].lstrip('#')[i:i+2], 16) / 255. for i in (0, 2, 4))
        overlay[pred_np > 0.5] = (r, g, b, PRED_ALPHA)
        ax.imshow(overlay, interpolation='nearest')
        ax.contour(pred_np, levels=[0.5], colors=[ACTION_COLOR[act]], linewidths=2.0)

        if is_agent:
            ax.text(0.5, 0.04, '▶  AGENT CHOICE',
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor=ACTION_COLOR[act],
                              edgecolor='none', alpha=0.95))

        n_steps = n_iters[act]
        style_ax(ax,
                 f'{ACTION_NAME[act]}  ({n_steps} steps)\nDice = {final_dice:.3f}',
                 title_color=ACTION_COLOR[act], highlight=is_agent)

    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor=GT_COLOR,
                       linestyle='--', linewidth=1.5, label='GT mask'),
    ] + [
        mpatches.Patch(facecolor=ACTION_COLOR[a], edgecolor=ACTION_COLOR[a],
                       alpha=0.7, label=f'{ACTION_NAME[a]} pred')
        for a in actions
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
               fontsize=8, frameon=False, labelcolor='white',
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'  ✓ Saved → {save_path}')


def make_figure(image, label, record, case_idx, save_path):
    """
    4-panel figure: [GT] [Point] [Box] [Scribble]
    The agent's chosen action is highlighted with a coloured border and star.
    """
    img_vol = image[0, 0].cpu() if image.ndim == 5 else image[0].cpu()
    lbl_vol = label[0, 0].cpu() if label.ndim == 5 else label[0].cpu()

    sl = best_slice(lbl_vol)
    img_np  = normalise_slice(img_vol[sl].numpy())
    lbl_np  = lbl_vol[sl].numpy()

    agent_action = record['agent_action']
    step_num     = record['step']
    actions      = [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]

    n_cols = 1 + len(actions)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.8 * n_cols, 4.2),
                             facecolor='#1a1a2e')
    fig.suptitle(
        f'Case {case_idx}  ·  Interaction step {step_num}'
        f'  ·  Agent chose: {ACTION_NAME[agent_action]}',
        color='white', fontsize=12, fontweight='bold', y=1.01
    )

    def style_ax(ax, title, title_color='white', highlight=False):
        ax.set_title(title, color=title_color, fontsize=10,
                     fontweight='bold' if highlight else 'normal', pad=6)
        ax.axis('off')
        if highlight:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(ACTION_COLOR[agent_action])
                spine.set_linewidth(5)

    # ── column 0: image + GT ───────────────────────────────────────────────
    ax = axes[0]
    ax.imshow(img_np, cmap='gray', interpolation='bilinear')
    ax.contour(lbl_np, levels=[0.5], colors=[GT_COLOR], linewidths=1.5)
    style_ax(ax, 'Image + GT', 'white')

    # ── columns 1-3: each action ──────────────────────────────────────────
    for col, act in enumerate(actions, start=1):
        mask_sig, dice = record['results'][act]
        if mask_sig.ndim == 5:
            pred_np = mask_sig[0, 0, sl].numpy()
        else:
            pred_np = mask_sig[0, sl].numpy()

        is_agent = (act == agent_action)
        ax = axes[col]
        ax.imshow(img_np, cmap='gray', interpolation='bilinear')

        # ground-truth contour
        ax.contour(lbl_np, levels=[0.5], colors=[GT_COLOR], linewidths=1.2, linestyles='--')

        # prediction filled overlay + contour
        overlay = np.zeros((*pred_np.shape, 4))
        c = plt.colormaps['tab10'](col - 1)
        overlay[pred_np > 0.5] = (*c[:3], PRED_ALPHA)
        ax.imshow(overlay, interpolation='nearest')
        ax.contour(pred_np, levels=[0.5], colors=[ACTION_COLOR[act]], linewidths=2.0)

        # Prominent "AGENT CHOICE" banner at the bottom of the panel
        if is_agent:
            ax.text(0.5, 0.04, '▶  AGENT CHOICE',
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor=ACTION_COLOR[act],
                              edgecolor='none', alpha=0.95))

        label_str = f'{ACTION_NAME[act]}'
        title     = f'{label_str}\nDice = {dice:.3f}'
        style_ax(ax, title, title_color=ACTION_COLOR[act], highlight=is_agent)

    # ── legend ─────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor=GT_COLOR,
                       linestyle='--', linewidth=1.5, label='GT mask'),
    ] + [
        mpatches.Patch(facecolor=ACTION_COLOR[a], edgecolor=ACTION_COLOR[a],
                       alpha=0.7, label=f'{ACTION_NAME[a]} pred')
        for a in actions
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
               fontsize=8, frameon=False, labelcolor='white',
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'  ✓ Saved → {save_path}')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Extend the shared config_args parser with viz-specific flags
    parser.add_argument('--case_idx',    type=int, default=0,
                        help='Index of test case to visualise (0-based).')
    parser.add_argument('--viz_step',    type=int, default=-1,
                        help='Interaction step to plot (1-based). '
                             '-1 = auto-select the most contrasting step.')
    parser.add_argument('--n_cases',     type=int, default=1,
                        help='Number of consecutive cases to visualise.')
    parser.add_argument('--viz_save_dir', type=str, default='./viz_prompt',
                        help='Output directory for saved figures (separate from checkpoint dir).')
    parser.add_argument('--auto_k', action='store_true',
                        help='Scan all K=1..iter_nums, run each action type independently at '
                             'every step, and auto-select the K where the agent\'s chosen action '
                             'most outperforms the alternatives (max advantage). '
                             'Overrides --fixed_k.')
    parser.add_argument('--compare_mode', type=str, default='final',
                        choices=['final', 'step'],
                        help='"final": three independent full runs compared side-by-side (recommended). '
                             '"step": per-step counterfactuals from the agent\'s sequential state.')
    parser.add_argument('--column_order', type=str, default=None,
                        help='Comma-separated column order, e.g. "scribble,box,point". '
                             'Default: point,box,scribble.')
    args = parser.parse_args()

    # Parse column_order into action constants
    _name_to_action = {'point': ACTION_POINT, 'box': ACTION_BOX, 'scribble': ACTION_SCRIBBLE}
    if args.column_order:
        args.column_order = [_name_to_action[n.strip().lower()] for n in args.column_order.split(',')]
    else:
        args.column_order = None

    # Build checkpoint path the same way test.py does, then let
    # check_and_setup_parser rewrite args.save_dir (we captured ckpt already).
    import logging
    from utils.util import setup_logger
    ckpt = os.path.join(args.save_dir, args.data, args.save_name,
                        args.checkpoint + '.pth.tar')
    check_and_setup_parser(args)   # modifies args.save_dir in-place

    os.makedirs(args.viz_save_dir, exist_ok=True)
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    print(f'Checkpoint : {ckpt}')
    print(f'Output dir : {args.viz_save_dir}')

    # ── load model ─────────────────────────────────────────────────────────
    sam  = build_model(args, checkpoint=ckpt)
    sam.image_encoder.eval()
    sam.prompt_encoder.eval()
    sam.mask_decoder.eval()

    edit_encoder = EditDeltaEncoder(in_channels=4).to(device)
    memory       = CorrectionPatternMemory(max_iters=args.iter_nums + 2).to(device)
    type_policy  = EditConditionedTypePolicy(
        state_dim=EditConditionedTypePolicy.STATE_DIM).to(device)

    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    if 'medsa_state' in ckpt_data:
        edit_encoder.load_state_dict(ckpt_data['medsa_state']['edit_encoder'])
        memory.load_state_dict(ckpt_data['medsa_state']['memory'])
        type_policy.load_state_dict(ckpt_data['medsa_state']['type_policy'])
        print('Loaded MedSA weights from checkpoint.')
    else:
        print('WARNING: no medsa_state in checkpoint — policy is random.')

    edit_encoder.eval()
    memory.eval()
    type_policy.eval()

    # ── load data ──────────────────────────────────────────────────────────
    val_loader = get_dataloader(args, split='test')
    dataset    = list(val_loader)   # materialise so we can index

    for offset in range(args.n_cases):
        idx = args.case_idx + offset
        if idx >= len(dataset):
            print(f'Case {idx} out of range ({len(dataset)} test cases). Stopping.')
            break

        sample = dataset[idx]
        # Handle all dataloader output formats:
        #   1. (image_tensor, label_tensor, ...)  — most CT datasets
        #   2. dict{'image': tensor, 'label': tensor}  — some TorchIO datasets
        #   3. (dict{'image': {'data': tensor}, 'label': {'data': tensor}}, ...) — lits/TorchIO sliding-window
        def _extract_tensor(v):
            """Unwrap TorchIO-style {'data': tensor_or_list} or plain tensor."""
            if isinstance(v, dict):
                d = v['data']
                # TorchIO batch collate returns a list of [C,D,H,W] MetaTensors;
                # unsqueeze to get [1,C,D,H,W] matching the expected batch format.
                if isinstance(d, (list, tuple)):
                    return d[0].unsqueeze(0)
                return d
            if isinstance(v, (list, tuple)):
                return v[0]
            return v

        if isinstance(sample, (list, tuple)) and isinstance(sample[0], dict):
            # TorchIO subject dict wrapped in tuple
            image_vol = _extract_tensor(sample[0]['image']).to(device)
            label_vol = _extract_tensor(sample[0]['label']).to(device)
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            image_vol = _extract_tensor(sample[0]).to(device)
            label_vol = _extract_tensor(sample[1]).to(device)
        elif isinstance(sample, dict):
            image_vol = _extract_tensor(sample.get('image', sample)).to(device)
            label_vol = _extract_tensor(sample.get('label', sample)).to(device)
        else:
            raise RuntimeError(f'Unexpected dataloader output format: {type(sample)}')

        # For large 3D volumes (LiTS-style) that exceed the model's patch size,
        # extract the 128³ patch with the most foreground voxels for visualization.
        PATCH = 128
        needs_patching = any(s > PATCH for s in image_vol.shape[2:])
        if needs_patching:
            import torchio as _tio
            subj = _tio.Subject(
                image=_tio.ScalarImage(tensor=image_vol[0].cpu().float()),
                label=_tio.LabelMap(tensor=label_vol[0].cpu().float()),
            )
            grid = _tio.inference.GridSampler(subj, PATCH, patch_overlap=16)
            best_patch_img, best_patch_lbl, best_fg = None, None, -1
            for pb in torch.utils.data.DataLoader(grid, batch_size=1):
                lbl_p = pb['label'][_tio.DATA]  # [1,1,P,P,P]
                fg = lbl_p.sum().item()
                if fg > best_fg:
                    best_fg = fg
                    best_patch_img = pb['image'][_tio.DATA].float().to(device)
                    best_patch_lbl = lbl_p.to(device)
            if best_patch_img is None or best_fg == 0:
                print(f'Case {idx}: no foreground patch found, skipping.')
                continue
            image, label = best_patch_img, best_patch_lbl
            print(f'  Sliding-window volume: selected patch with {int(best_fg)} fg voxels.')
        else:
            image, label = image_vol, label_vol

        # Skip multi-channel images (some kits cases have unexpected 2-ch data)
        if image.shape[1] != 1:
            print(f'Case {idx}: skipping — unexpected image channels ({image.shape[1]}).')
            continue

        if label.sum() == 0:
            print(f'Case {idx}: empty label, skipping.')
            continue

        print(f'\n── Case {idx} ──────────────────────────────────────────')

        if args.compare_mode == 'final':
            # ── independent full run for each action type ──────────────────
            # First: let the agent decide how many steps it would use
            with torch.no_grad():
                step_records = interaction_with_counterfactuals(
                    sam, edit_encoder, memory, type_policy, image, label, args
                )
            n_agent_steps = len(step_records)
            agent_action  = step_records[-1]['agent_action'] if step_records else ACTION_POINT

            max_k = args.iter_nums

            if args.auto_k:
                # Run all three action types for the full budget, collect per-step Dice
                print(f'  Scanning K=1..{max_k} for best agent advantage ...')
                all_curves = {}
                all_masks_at_k = {}  # {act: [mask_at_k1, mask_at_k2, ...]}
                for act in [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]:
                    with torch.no_grad():
                        mask_sig, _, step_dices = run_fixed_action(
                            sam, image, label, act, max_k, args)
                    all_curves[act] = step_dices
                    # re-run collecting per-step masks
                    dev = args.device
                    img_emb, feat = sam.image_encoder(image)
                    prev = torch.zeros_like(label).to(dev)
                    masks_per_step = []
                    for it in range(max_k):
                        ms, _ = forward_one_action(sam, img_emb, feat, prev, label, act, args)
                        masks_per_step.append(ms)
                        prev = torch.logit(ms.to(dev).clamp(1e-6, 1-1e-6))
                    all_masks_at_k[act] = masks_per_step

                    print(f'  {ACTION_NAME[act]:8s}: '
                          + ' '.join(f'{d:.3f}' for d in step_dices))

                # Find K where agent_action Dice - max(other Dice) is largest
                best_k, best_adv = 1, -999.0
                for k in range(max_k):
                    agent_d = all_curves[agent_action][k]
                    other_d = max(all_curves[a][k]
                                  for a in [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]
                                  if a != agent_action)
                    adv = agent_d - other_d
                    if adv > best_adv:
                        best_adv, best_k = adv, k + 1   # 1-based

                print(f'  Auto-K={best_k} (agent advantage = {best_adv:+.3f})')
                n_budget = best_k
                results_by_action = {
                    act: (all_masks_at_k[act][best_k-1],
                          all_curves[act][best_k-1],
                          all_curves[act][:best_k])
                    for act in [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]
                }

            else:
                # Simple fixed/agent budget
                n_budget = args.fixed_k if args.fixed_k > 0 else n_agent_steps
                print(f'  Budget: {n_budget} steps '
                      f'({"user-fixed" if args.fixed_k > 0 else f"agent stopped at {n_agent_steps}"})')

                results_by_action = {}
                for act in [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]:
                    with torch.no_grad():
                        mask_sig, final_dice, step_dices = run_fixed_action(
                            sam, image, label, act, n_budget, args)
                    results_by_action[act] = (mask_sig, final_dice, step_dices)
                    print(f'  {ACTION_NAME[act]:8s} ({n_budget} steps): '
                          + ' '.join(f'{d:.3f}' for d in step_dices))

            save_path = os.path.join(
                args.viz_save_dir,
                f'case{idx:03d}_K{n_budget}_{args.data}.png'
            )
            col_order = getattr(args, 'column_order', None)
            make_final_figure(image.cpu(), label.cpu(), results_by_action,
                              agent_action, n_budget, idx, save_path,
                              column_order=col_order)

        else:
            # ── per-step counterfactuals from agent's sequential state ─────
            with torch.no_grad():
                step_records = interaction_with_counterfactuals(
                    sam, edit_encoder, memory, type_policy, image, label, args
                )
            if not step_records:
                print('  No interaction steps recorded.')
                continue

            if args.viz_step > 0:
                record = next((r for r in step_records if r['step'] == args.viz_step),
                              step_records[-1])
            else:
                def advantage(rec):
                    agent_dice = rec['results'][rec['agent_action']][1]
                    alts = [rec['results'][a][1]
                            for a in [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]
                            if a != rec['agent_action']]
                    return agent_dice - max(alts) if alts else 0.0
                record = max(step_records, key=advantage)
                print(f'  Auto-selected step {record["step"]} '
                      f'(agent advantage = {advantage(record):+.3f})')

            save_path = os.path.join(
                args.viz_save_dir,
                f'case{idx:03d}_step{record["step"]}_{args.data}.png'
            )
            make_figure(image.cpu(), label.cpu(), record, idx, save_path)


if __name__ == '__main__':
    main()
