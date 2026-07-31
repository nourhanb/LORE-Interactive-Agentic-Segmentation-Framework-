"""
plot_efficiency_curve.py

For each test case, runs three independent fixed-action runs (point, box, scribble)
for iter_nums steps and records Dice at every step.  Also runs the CoCC agent
(adaptive, policy-driven) and records its stopping step and final Dice.

Plots mean Dice vs. interaction step for all three fixed-action baselines,
and marks the agent's operating point (K-bar, mean Dice) as a star.

Usage:
    python src-agentic/plot_efficiency_curve.py \
        --data pancreas  [same flags as test.py]  \
        --viz_save_dir ./viz_prompt/pancreas
"""

import os, math, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config.config_args import *
from config.config_setup import build_model, get_dataloader
from utils.util import _bbox_mask
from utils import scribble
from models.medsa import EditDeltaEncoder, CorrectionPatternMemory
from models.policy import (
    EditConditionedTypePolicy,
    ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE, ACTION_STOP,
)
from processor.trainer import compute_geometric_descriptors, _ensure_5d

ACTION_NAME  = {ACTION_POINT: 'Point', ACTION_BOX: 'Box', ACTION_SCRIBBLE: 'Scribble'}
ACTION_COLOR = {ACTION_POINT: '#2196F3', ACTION_BOX: '#FF9800', ACTION_SCRIBBLE: '#4CAF50'}


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def dice_coef(pred, label):
    pred_bin = (pred > 0.5).float()
    tp = (pred_bin * label).sum()
    return (2 * tp / (pred_bin.sum() + label.sum() + 1e-6)).item()


def forward_one_action(sam, image_embedding, feature_list, prev_masks, label, action, args):
    dev = args.device
    prev_sig = torch.sigmoid(prev_masks)
    pred_bin = (prev_sig > 0.5);  true_bin = (label > 0)
    fn_masks = (true_bin & ~pred_bin);  fp_masks = (~true_bin & pred_bin)

    bp_list, bl_list = [], []
    if action == ACTION_SCRIBBLE and getattr(args, 'use_scribble', False):
        error_vol = (fn_masks[0].float() + fp_masks[0].float()).clamp(0, 1)
        k_top     = max(1, int(error_vol.numel() * 0.10))
        threshold = error_vol.flatten().topk(k=k_top).values.min()
        scrib_obj = scribble.CenterlineScribble()
        for region, polarity in [
            (fn_masks[0].float() * (error_vol >= threshold).float(), 1),
            (fp_masks[0].float() * (error_vol >= threshold).float(), 0),
        ]:
            if region.sum() == 0: continue
            sm = scrib_obj.batch_scribble(region.permute(3,0,1,2)).permute(1,2,3,0) > 0
            coords = torch.argwhere(sm)[:,1:].unsqueeze(0)
            if coords.numel() > 0:
                coords = coords[:, :min(coords.size(1), 500), :]
                bl_list.append(torch.full((1, coords.size(1)), float(polarity)))
                bp_list.append(coords)

    if not bp_list:
        to_pt = fn_masks[0] | fp_masks[0]
        pts   = torch.argwhere(to_pt)
        if len(pts) == 0: pts = torch.argwhere(true_bin[0])
        if len(pts) == 0: pts = label.new_zeros(1, 4, dtype=torch.long)
        n_cl  = max(min(getattr(args, 'num_clicks_validation', 1), len(pts)), 1)
        sel   = pts[np.random.choice(len(pts), n_cl, replace=False)]
        for ci in range(len(sel)):
            pt = sel[ci]; is_pos = bool(fn_masks[0,0,pt[1],pt[2],pt[3]])
            bp_list.append(pt[1:].clone().detach().reshape(1,1,3))
            bl_list.append(torch.tensor([float(int(is_pos))]).reshape(1,1))

    pts_co = torch.cat(bp_list, dim=1).to(dev)
    pts_la = torch.cat(bl_list, dim=1).to(dev)
    # Match test.py get_points: box cue for BOX and POINT actions (not SCRIBBLE)
    bbox   = _bbox_mask(label[:,0,:]).to(dev) if (args.use_box and action != ACTION_SCRIBBLE) else None

    prev_down = F.interpolate(prev_masks, scale_factor=0.25)
    feat_dev  = [f.to(dev) for f in feature_list]
    pt_emb, img_emb = sam.prompt_encoder(
        points=[pts_co, pts_la], boxes=bbox,
        masks=prev_down, image_embeddings=image_embedding.to(dev))
    mask, pred_dice_sc, _, _ = sam.mask_decoder(
        prompt_embeddings=pt_emb, image_embeddings=img_emb, feature_list=feat_dev)
    if getattr(args, 'multiple_outputs', False):
        _, idx = torch.max(pred_dice_sc, dim=1)
        mask   = mask[torch.arange(mask.size(0)), idx].unsqueeze(1)

    mask_sig = torch.sigmoid(mask).detach().cpu()
    return mask_sig, dice_coef(mask_sig, label.cpu())


def run_fixed_action_curve(sam, image, label, action, n_iters, args):
    """Returns list of Dice values at each of the n_iters steps."""
    dev = args.device
    image_embedding, feature_list = sam.image_encoder(image)
    prev_masks = torch.zeros_like(label).to(dev)
    step_dices = []
    for it in range(n_iters):
        mask_sig, dice = forward_one_action(
            sam, image_embedding, feature_list, prev_masks, label, action, args)
        step_dices.append(dice)
        prev_masks = torch.logit(mask_sig.to(dev).clamp(1e-6, 1-1e-6))
    return step_dices


def run_agent(sam, edit_encoder, memory, type_policy, image, label, args):
    """Run the CoCC agent (policy-driven stopping). Returns (n_steps_used, final_dice)."""
    dev = args.device; max_iters = args.iter_nums
    image_embedding, feature_list = sam.image_encoder(image)
    prev_masks = torch.zeros_like(label).to(dev)
    memory.reset()
    e_full = torch.zeros(image.size(0), 132, device=dev)
    m_i    = torch.zeros(image.size(0), 132, device=dev)
    prev_dice = 0.0; prev_hmap = None; n_steps = 0

    for it in range(max_iters):
        prev_sig = torch.sigmoid(prev_masks) if it > 0 else prev_masks
        action = ACTION_POINT
        if it > 0:
            l5 = _ensure_5d(label); p5 = _ensure_5d(prev_sig); i5 = _ensure_5d(image)
            pb = (p5 > 0.5); tb = (l5 > 0)
            fn = (tb & ~pb).float(); fp = (~tb & pb).float()
            with torch.no_grad():
                geo  = compute_geometric_descriptors(fn, fp, p5)
                e_full = edit_encoder(fn, fp, pb.float(), i5, geo)
                m_i, _ = memory.update(e_full)
                e_cnn = e_full[:,:128]; m_cnn = m_i[:,:128]
                e_n   = torch.norm(e_cnn, dim=1).mean().item()
                e_mag = e_n / (e_n + 1.0)
                pers  = float(F.cosine_similarity(e_cnn, m_cnn, dim=1).mean().item())
                if prev_hmap is not None:
                    h = torch.sigmoid(prev_hmap).flatten().float()
                    h = h / (h.sum() + 1e-8)
                    ent = -(h * torch.log(h + 1e-10)).sum().item()
                    sent = ent / math.log(float(h.numel()) + 1.0)
                else:
                    sent = 1.0
                cd = dice_coef(prev_sig.cpu(), label.cpu())
                state = EditConditionedTypePolicy.build_state(
                    dice_current=cd, delta_dice=cd-prev_dice,
                    iter_progress=it/max(max_iters,1),
                    edit_volume=float(geo[0,0].item()),
                    edit_bnd_ovlp=float(geo[0,3].item()),
                    error_magnitude=e_mag, persistence=pers,
                    spatial_entropy=sent, device=dev)
                action = type_policy.select_action(
                    state, epsilon=0.0,
                    edit_volume=float(geo[0,0].item()),
                    iter_idx=it, max_iters=max_iters, dice_current=cd)
                prev_dice = cd
            if action == ACTION_STOP:
                break

        mask_sig, _ = forward_one_action(
            sam, image_embedding, feature_list, prev_masks, label, action, args)
        prev_masks = torch.logit(mask_sig.to(dev).clamp(1e-6, 1-1e-6))
        n_steps += 1

    final_dice = dice_coef(torch.sigmoid(prev_masks).cpu(), label.cpu())
    return n_steps, final_dice


def make_efficiency_plot(curves, agent_k_list, agent_dice_list, dataset_name, save_path):
    """
    curves: {action: [mean_dice_step1, ..., mean_dice_step11]}
    agent_k_list, agent_dice_list: per-case (K_used, final_dice)
    """
    n_steps = max(len(v) for v in curves.values())
    steps   = list(range(1, n_steps + 1))

    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor='white')

    # Fixed-action curves
    line_styles = {ACTION_POINT: '-', ACTION_BOX: '--', ACTION_SCRIBBLE: '-.'}
    for act in [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]:
        mean_c = [np.mean(curves[act][k]) for k in range(n_steps)]
        std_c  = [np.std(curves[act][k])  for k in range(n_steps)]
        ax.plot(steps, [d*100 for d in mean_c],
                color=ACTION_COLOR[act], lw=2,
                linestyle=line_styles[act],
                label=f'Fixed {ACTION_NAME[act]}', zorder=3)
        ax.fill_between(steps,
                        [(d-s)*100 for d,s in zip(mean_c, std_c)],
                        [(d+s)*100 for d,s in zip(mean_c, std_c)],
                        color=ACTION_COLOR[act], alpha=0.12, zorder=2)

    # Agent operating point
    agent_k_mean  = np.mean(agent_k_list)
    agent_d_mean  = np.mean(agent_dice_list) * 100
    ax.scatter([agent_k_mean], [agent_d_mean],
               marker='*', s=280, color='crimson', zorder=5,
               label=f'CoCC Agent  (K̄={agent_k_mean:.1f}, Dice={agent_d_mean:.1f}%)')
    ax.axvline(agent_k_mean, color='crimson', lw=1.2, linestyle=':', alpha=0.7, zorder=4)

    ax.set_xlabel('Interaction steps (K)', fontsize=12)
    ax.set_ylabel('Mean Dice (%)', fontsize=12)
    ax.set_title(f'Interaction Efficiency — {dataset_name.capitalize()}', fontsize=13, fontweight='bold')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(0.5, n_steps + 0.5)
    ax.legend(fontsize=9.5, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved → {save_path}')


def main():
    parser.add_argument('--viz_save_dir', type=str, default='./viz_prompt',
                        help='Output directory for the efficiency curve figure.')
    args = parser.parse_args()

    ckpt = os.path.join(args.save_dir, args.data, args.save_name,
                        args.checkpoint + '.pth.tar')
    check_and_setup_parser(args)
    os.makedirs(args.viz_save_dir, exist_ok=True)
    set_seed(42)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Checkpoint : {ckpt}')

    sam = build_model(args, checkpoint=ckpt)
    sam.image_encoder.eval(); sam.prompt_encoder.eval(); sam.mask_decoder.eval()

    edit_encoder = EditDeltaEncoder(in_channels=4).to(args.device)
    memory       = CorrectionPatternMemory(max_iters=args.iter_nums+2).to(args.device)
    type_policy  = EditConditionedTypePolicy(
        state_dim=EditConditionedTypePolicy.STATE_DIM).to(args.device)

    ckpt_data = torch.load(ckpt, map_location=args.device, weights_only=False)
    if 'medsa_state' in ckpt_data:
        edit_encoder.load_state_dict(ckpt_data['medsa_state']['edit_encoder'])
        memory.load_state_dict(ckpt_data['medsa_state']['memory'])
        type_policy.load_state_dict(ckpt_data['medsa_state']['type_policy'])
        print('Loaded MedSA weights.')
    edit_encoder.eval(); memory.eval(); type_policy.eval()

    val_loader = get_dataloader(args, split='test')
    n_steps    = args.iter_nums

    # per-step Dice lists: curves[action][step_idx] = [case_dice, ...]
    curves = {
        ACTION_POINT:    [[] for _ in range(n_steps)],
        ACTION_BOX:      [[] for _ in range(n_steps)],
        ACTION_SCRIBBLE: [[] for _ in range(n_steps)],
    }
    agent_k_list, agent_dice_list = [], []

    for idx, sample in enumerate(val_loader):
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            image, label = sample[0].to(args.device), sample[1].to(args.device)
        else:
            raise RuntimeError('Unexpected dataloader output.')
        if label.sum() == 0:
            print(f'  Case {idx}: empty label, skipping.'); continue

        print(f'  Case {idx+1}/{len(val_loader)} ...', end=' ', flush=True)

        with torch.no_grad():
            # agent run
            k, d = run_agent(sam, edit_encoder, memory, type_policy, image, label, args)
            agent_k_list.append(k); agent_dice_list.append(d)

            # fixed-action runs
            for act in [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]:
                step_dices = run_fixed_action_curve(sam, image, label, act, n_steps, args)
                for si, dice in enumerate(step_dices):
                    curves[act][si].append(dice)

        print(f'agent K={k} d={d:.3f}')

    # plot
    save_path = os.path.join(args.viz_save_dir, f'efficiency_curve_{args.data}.png')
    make_efficiency_plot(curves, agent_k_list, agent_dice_list, args.data, save_path)

    # print summary
    print(f'\n=== Summary ({args.data}) ===')
    print(f'Agent:    K̄={np.mean(agent_k_list):.2f}  Dice={np.mean(agent_dice_list)*100:.2f}%')
    for act in [ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE]:
        final_dices = curves[act][-1]
        print(f'{ACTION_NAME[act]:8s}: Dice@{n_steps}={np.mean(final_dices)*100:.2f}%  '
              f'Dice@7={np.mean(curves[act][6])*100:.2f}%')


if __name__ == '__main__':
    main()
