"""
visualize_cross_anatomy.py

For each specified dataset, run the CoCC agent on one representative test case
and display the final segmentation result.  All datasets are shown side-by-side
in one publication-quality figure.

Usage
-----
  python visualize_cross_anatomy.py \\
      --datasets kits pancreas colon autopet \\
      --save_dir ./implementation_medsa \\
      --out_path ./viz_prompt/cross_anatomy.png \\
      --n_scan 15 \\          # cases scanned per dataset to find the best one
      [all standard SAM/MedSA flags]
"""

import os, sys, math, random, copy, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── import from existing visualise script ────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config.config_args import *          # imports module-level `parser`
from config.config_setup import build_model, get_dataloader
from utils.util import _bbox_mask
from models.medsa import EditDeltaEncoder, CorrectionPatternMemory
from models.policy import (
    EditConditionedTypePolicy,
    ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE, ACTION_STOP,
)
from processor.trainer import compute_geometric_descriptors, _ensure_5d
from visualize_prompt_comparison import (
    forward_one_action, best_slice, normalise_slice, dice_coef, set_seed,
    ACTION_NAME, ACTION_COLOR,
)

# ── per-dataset metadata ──────────────────────────────────────────────────────
ORGAN_META = {
    'kits':     {'display': 'Kidney',   'save_name': 'kits_medsa',     'color': '#E91E63'},
    'pancreas': {'display': 'Pancreas', 'save_name': 'pancreas_medsa', 'color': '#29B6F6'},
    'autopet':  {'display': 'Lung',     'save_name': 'autopet_medsa',  'color': '#FF9800'},
    'colon':    {'display': 'Colon',    'save_name': 'colon_medsa',    'color': '#66BB6A'},
    'lits':     {'display': 'Liver',    'save_name': 'lits_medsa',     'color': '#AB47BC'},
    'hecktor':  {'display': 'H&N',      'save_name': 'hecktor_medsa',  'color': '#FF7043'},
}

GT_COLOR   = 'white'
PRED_ALPHA = 0.45
FIG_BG     = '#111111'


# ── agent runner (no counterfactuals) ────────────────────────────────────────

def run_agent_only(sam, edit_encoder, memory, type_policy, image, label, args):
    """
    Run the CoCC agent until it stops or exhausts iter_nums.
    Returns (final_mask_sig_cpu, K, actions_list, per_step_dice).
    """
    dev       = args.device
    max_iters = args.iter_nums
    image_embedding, feature_list = sam.image_encoder(image)

    prev_masks        = torch.zeros_like(label).to(dev)
    memory.reset()
    e_full            = torch.zeros(image.size(0), 132, device=dev)
    m_i               = torch.zeros(image.size(0), 132, device=dev)
    prev_dice         = 0.0
    prev_spatial_hmap = None
    actions           = []
    step_dices        = []

    for iter_num in range(max_iters):
        prev_sig = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

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
                e_cnn  = e_full[:, :128];  m_cnn = m_i[:, :128]
                e_norm = torch.norm(e_cnn, dim=1).mean().item()
                e_mag  = e_norm / (e_norm + 1.0)
                persistence = float(F.cosine_similarity(e_cnn, m_cnn, dim=1).mean().item())

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
            break

        with torch.no_grad():
            mask_sig, dice = forward_one_action(
                sam, image_embedding, feature_list, prev_masks, label, action, args
            )
        actions.append(action)
        step_dices.append(dice)
        prev_masks = torch.logit(mask_sig.to(dev).clamp(1e-6, 1 - 1e-6))

    final_mask = mask_sig if actions else torch.sigmoid(prev_masks).cpu()
    return final_mask, len(actions), actions, step_dices


# ── per-dataset pipeline ──────────────────────────────────────────────────────

def process_dataset(dataset_key, args_base, case_indices, n_scan):
    """
    Load the checkpoint for `dataset_key`, scan up to n_scan test cases,
    and return the one with the highest final Dice.
    """
    meta = ORGAN_META[dataset_key]
    args = copy.deepcopy(args_base)
    args.data      = dataset_key
    args.save_name = meta['save_name']
    args.data_dir  = f"src-agentic/splits/{dataset_key}"

    ckpt_path = os.path.join(args.save_dir, args.data, args.save_name, 'best.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.save_dir, args.data, args.save_name, 'best.pth.tar')
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] checkpoint not found: {ckpt_path}")
        return None

    print(f"  Loading {ckpt_path}")
    sam = build_model(args, checkpoint=ckpt_path).to(args.device)
    sam.eval()

    # MedSA components
    edit_encoder = EditDeltaEncoder(in_channels=4).to(args.device)
    memory = CorrectionPatternMemory(max_iters=args.iter_nums + 2).to(args.device)
    type_policy = EditConditionedTypePolicy(
        state_dim=EditConditionedTypePolicy.STATE_DIM
    ).to(args.device)

    # Load MedSA weights from checkpoint
    ckpt_data = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    if 'medsa_state' in ckpt_data:
        edit_encoder.load_state_dict(ckpt_data['medsa_state']['edit_encoder'])
        memory.load_state_dict(ckpt_data['medsa_state']['memory'])
        type_policy.load_state_dict(ckpt_data['medsa_state']['type_policy'])
        print('  Loaded MedSA weights.')
    else:
        print('  WARNING: no medsa_state in checkpoint — policy is random.')
    edit_encoder.eval(); memory.eval(); type_policy.eval()

    # Data loader — materialise to allow indexing
    loader  = get_dataloader(args, split=args.split)
    dataset = list(loader)
    best    = None

    explicit = case_indices.get(dataset_key)
    indices  = [explicit] if explicit is not None else list(range(min(n_scan, len(dataset))))

    for ci in indices:
        if ci >= len(dataset):
            break
        sample = dataset[ci]
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            image, label = sample[0].to(args.device), sample[1].to(args.device)
        else:
            image = sample['image'].to(args.device)
            label = sample['label'].to(args.device)

        if label.sum() == 0:
            print(f"  case {ci}: empty label, skip")
            continue

        set_seed(42)
        with torch.no_grad():
            final_mask, K, actions, step_dices = run_agent_only(
                sam, edit_encoder, memory, type_policy, image, label, args
            )
        final_dice = step_dices[-1] if step_dices else 0.0
        print(f"  case {ci:3d}: K={K:2d}  Dice={final_dice:.3f}  "
              f"actions=[{''.join(ACTION_NAME[a][0] for a in actions)}]")

        if best is None or final_dice > best['dice']:
            best = dict(
                image=image.cpu(), label=label.cpu(),
                mask=final_mask.cpu(), K=K, actions=actions,
                dice=final_dice, case_idx=ci,
            )

    # Free GPU memory before next dataset
    del sam, edit_encoder, memory, type_policy
    torch.cuda.empty_cache()

    if best:
        print(f"  → best: case {best['case_idx']}  K={best['K']}  Dice={best['dice']:.3f}")
    return best, meta


# ── figure builder ────────────────────────────────────────────────────────────

def make_cross_anatomy_figure(panels, out_path):
    """
    panels: list of dicts with keys image_np, gt_np, pred_np, organ,
            dice, K, actions, color
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.8),
                             facecolor=FIG_BG, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, p in zip(axes, panels):
        ax.set_facecolor(FIG_BG)
        ax.imshow(p['image_np'], cmap='gray', interpolation='bilinear', aspect='equal')

        # Prediction fill
        pred_bin = (p['pred_np'] > 0.5).astype(float)
        r, g, b = tuple(int(p['color'].lstrip('#')[i:i+2], 16) / 255. for i in (0, 2, 4))
        overlay = np.zeros((*pred_bin.shape, 4))
        overlay[..., 0] = r; overlay[..., 1] = g; overlay[..., 2] = b
        overlay[..., 3] = pred_bin * PRED_ALPHA
        ax.imshow(overlay, interpolation='nearest', aspect='equal')

        # GT contour
        ax.contour(p['gt_np'], levels=[0.5], colors=[GT_COLOR], linewidths=2.0, alpha=0.9)

        # Action strip at bottom
        action_str = '  '.join(ACTION_NAME[a][0] for a in p['actions'])  # P P B B …
        ax.text(0.5, 0.03, action_str,
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=7.5, color='lightgray',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#222', alpha=0.7))

        # Title
        ax.set_title(
            f"{p['organ']}\nDice={p['dice']:.1%}   K={p['K']}",
            color='white', fontsize=10, fontweight='bold', pad=6,
        )
        ax.axis('off')

        # Coloured border for the organ
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(p['color'])
            spine.set_linewidth(3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor=GT_COLOR, linewidth=2,
                       label='Ground truth'),
        mpatches.Patch(facecolor='#2196F3', alpha=0.5, label='Agent prediction'),
        mpatches.Patch(facecolor='none', edgecolor='none',
                       label='K = no. of interactions'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               framealpha=0.5, facecolor='#222', labelcolor='white',
               fontsize=8, bbox_to_anchor=(0.5, -0.04))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=FIG_BG)
    print(f"\n✓ Saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser.add_argument('--datasets',     nargs='+',
                        default=['kits', 'pancreas', 'colon', 'autopet'])
    parser.add_argument('--out_path',     type=str,
                        default='./viz_prompt/cross_anatomy.png')
    parser.add_argument('--n_scan',       type=int, default=15,
                        help='Max test cases to scan per dataset to find the best one')
    parser.add_argument('--case_indices', nargs='*', type=str, default=[],
                        help='dataset:idx pairs, e.g. kits:2 pancreas:16. '
                             'Overrides auto-scan for that dataset.')
    args = parser.parse_args()

    set_seed(42)

    # Parse explicit case indices
    explicit_cases = {}
    for item in args.case_indices:
        ds, idx = item.split(':')
        explicit_cases[ds] = int(idx)

    panels = []
    for dataset_key in args.datasets:
        if dataset_key not in ORGAN_META:
            print(f"Unknown dataset: {dataset_key}, skipping.")
            continue
        print(f"\n{'═'*55}")
        print(f"  Dataset: {ORGAN_META[dataset_key]['display']} ({dataset_key})")
        print(f"{'═'*55}")

        result = process_dataset(dataset_key, args, explicit_cases, args.n_scan)
        if result is None:
            continue
        best, meta = result
        if best is None:
            print(f"  No valid result for {dataset_key}")
            continue

        # Extract best 2D slice
        img_vol = best['image'][0, 0]   # H W D
        lbl_vol = best['label'][0, 0]
        msk_vol = best['mask'][0, 0]

        sl = best_slice(lbl_vol)
        panels.append(dict(
            image_np = normalise_slice(img_vol[..., sl].numpy()),
            gt_np    = lbl_vol[..., sl].numpy(),
            pred_np  = msk_vol[..., sl].numpy(),
            organ    = meta['display'],
            dice     = best['dice'],
            K        = best['K'],
            actions  = best['actions'],
            color    = meta['color'],
        ))

    if not panels:
        print("\nNo panels generated — check checkpoints and data paths.")
        return

    make_cross_anatomy_figure(panels, args.out_path)


if __name__ == '__main__':
    main()
