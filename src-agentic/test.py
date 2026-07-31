import json
import logging
import math
import os
import os.path
import csv
import torch
from utils.util import setup_logger
from config.config_args import *
from config.config_setup import build_model, get_dataloader
import numpy as np
from torch.backends import cudnn
import time, random
import torch.nn.functional as F
from utils.util import _bbox_mask
from utils import scribble, boundary_selection
import torchio as tio
import surface_distance
from surface_distance import metrics
from models.medsa import EditDeltaEncoder, CorrectionPatternMemory, compute_geometric_descriptors
from models.policy import (
    EditConditionedTypePolicy,
    ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE, ACTION_STOP,
)
from models.cpc_uga import PriorityUncertaintyPolicy, mask_uncertainty
from processor.trainer_basic import _subject_from_dict, _tio_volume_4d, _pad_subject_for_grid
from processor.trainer import _ensure_5d
from utils.clinical_priority import priority_mask_from_errors

# Fixed interaction budgets reported for paper curves / tables
_BUDGET_STEPS = (1, 3, 5, 11)
_ACT_NAME = {
    ACTION_POINT: 'point',
    ACTION_BOX: 'box',
    ACTION_SCRIBBLE: 'scribble',
    ACTION_STOP: 'stop',
}

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


class Tester(object):
    def __init__(self, args, logger, ckpt):
        self.args = args
        self.logger = logger

        self.val_data = get_dataloader(args, split='test')

        print('loading models and setting up')
        self.sam = build_model(args, checkpoint=ckpt)

        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder
        self.mask_decoder = self.sam.mask_decoder

        # Load agent components if enabled
        self.use_medsa = getattr(args, 'use_medsa', False)
        self.use_cpc_uga = getattr(args, 'use_cpc_uga', False)
        if self.use_cpc_uga and self.use_medsa:
            raise ValueError('--use_cpc_uga and --use_medsa are mutually exclusive')

        if self.use_cpc_uga:
            self.type_policy = PriorityUncertaintyPolicy(
                state_dim=PriorityUncertaintyPolicy.STATE_DIM,
            ).to(args.device)
            ckpt_data = torch.load(ckpt, map_location=args.device, weights_only=False)
            if 'cpc_uga_state' in ckpt_data:
                self.type_policy.load_state_dict(ckpt_data['cpc_uga_state']['type_policy'])
                logger.info("Loaded CPC–UGA policy weights from checkpoint.")
            else:
                logger.warning("CPC–UGA enabled but checkpoint has no cpc_uga_state — random policy!")
            self.type_policy.eval()

        elif self.use_medsa:
            self.edit_encoder = EditDeltaEncoder(in_channels=4).to(args.device)
            self.memory       = CorrectionPatternMemory(max_iters=args.iter_nums + 2).to(args.device)
            self.type_policy  = EditConditionedTypePolicy(
                state_dim=EditConditionedTypePolicy.STATE_DIM,
            ).to(args.device)
            # Load trained weights from checkpoint
            ckpt_data = torch.load(ckpt, map_location=args.device, weights_only=False)
            if 'medsa_state' in ckpt_data:
                self.edit_encoder.load_state_dict(ckpt_data['medsa_state']['edit_encoder'])
                self.memory.load_state_dict(ckpt_data['medsa_state']['memory'])
                self.type_policy.load_state_dict(ckpt_data['medsa_state']['type_policy'])
                logger.info("Loaded MedSA component weights from checkpoint.")
            else:
                logger.warning("MedSA enabled but checkpoint has no medsa_state — policy uses random weights!")
            self.edit_encoder.eval()
            self.memory.eval()
            self.type_policy.eval()

        # Paper-facing report directory (CSV + JSON)
        self.report_dir = os.path.join(args.save_dir, 'test_reports')
        os.makedirs(self.report_dir, exist_ok=True)

    @staticmethod
    def _empty_trace(max_iters: int = 11):
        return {
            'dice_curve': [],
            'uncert_curve': [],
            'prio_vol_curve': [],
            'action_seq': [],
            'stopped_early': False,
            'hit_max_iters': True,
            'accept_uncert': float('nan'),
            'mean_uncert': float('nan'),
            'final_prio_vol': float('nan'),
            'n_stop': 0,
        }

    def _surface_metrics(self, pred_logits, label, spacing_mm=(1.0, 1.0, 1.0)):
        """NSD@5mm and HD95 from a single predicted volume (logits or probs)."""
        if torch.is_tensor(pred_logits):
            pred_bin = (torch.sigmoid(pred_logits) > 0.5).detach().cpu().numpy()
        else:
            pred_bin = (pred_logits > 0.5)
        if torch.is_tensor(label):
            gt_bin = (label > 0).detach().cpu().numpy()
        else:
            gt_bin = (label > 0)

        # Squeeze to 3D
        while pred_bin.ndim > 3:
            pred_bin = pred_bin[0]
        while gt_bin.ndim > 3:
            gt_bin = gt_bin[0]
        pred_bin = pred_bin.astype(bool)
        gt_bin = gt_bin.astype(bool)

        if not gt_bin.any() and not pred_bin.any():
            return 1.0, 0.0
        if not gt_bin.any() or not pred_bin.any():
            # Empty vs non-empty: NSD=0, HD95 = large sentinel
            return 0.0, float('nan')

        ssd = surface_distance.compute_surface_distances(gt_bin, pred_bin, spacing_mm=spacing_mm)
        nsd = float(metrics.compute_surface_dice_at_tolerance(ssd, 5))
        hd95 = float(metrics.compute_robust_hausdorff(ssd, 95))
        return nsd, hd95

    def _subject_id(self, image_path) -> str:
        if isinstance(image_path, (list, tuple)):
            image_path = image_path[0] if image_path else 'unknown'
        path = str(image_path)
        return os.path.splitext(os.path.basename(path))[0]

    def _load_pretrain_model(self, ckpt):
        model_dict = torch.load(ckpt, map_location=self.args.device)
        state_dict = model_dict
        self.sam.load_state_dict(state_dict['model_state_dict'])

    def validate(self, epoch_num):
        self.image_encoder.eval()
        self.prompt_encoder.eval()
        self.mask_decoder.eval()

        if use_sliding_window(self.args):
            loss = self.validater_sliding_window(epoch_num)
        else:
            loss = self.validater(epoch_num)
        return loss


    def validater_sliding_window(self, epoch_num):
        with torch.no_grad():
            case_rows = []
            dice_summary, nsd_summary, hd95_summary = [], [], []
            iters_summary, ces_summary, effort_summary = [], [], []
            act_pt_summary, act_bx_summary, act_sc_summary, act_st_summary = [], [], [], []
            uncert_accept, uncert_mean, early_stop_flags = [], [], []
            dice_at = {k: [] for k in _BUDGET_STEPS}
            nsd_at = {k: [] for k in _BUDGET_STEPS}
            dice_curve_all = []  # list of per-case curves (padded to max_iters)

            for idx, (subject_dict, image_path, subject_dict_save) in enumerate(self.val_data):
                label_vol = _tio_volume_4d(subject_dict['label']['data'][0])
                if label_vol.sum() <= 0:
                    self.logger.info(f'{image_path} label volume too small, skipped')
                    continue
                subject = _pad_subject_for_grid(_subject_from_dict(subject_dict), patch_size=128)
                grid_sampler = tio.inference.GridSampler(subject, 128, 16)
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
                aggregator   = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

                patch_iters, patch_effort = [], []
                all_counts = {'point': 0, 'box': 0, 'scribble': 0, 'stop': 0}
                patch_traces = []
                for idx_patch, patches_batch in enumerate(patch_loader):
                    image, label = (patches_batch['image'][tio.DATA].to(self.args.device),
                                    patches_batch['label'][tio.DATA].to(self.args.device))
                    locations = patches_batch[tio.LOCATION]

                    if torch.count_nonzero(label) == 0:
                        masks = torch.zeros([1, 1, 128, 128, 128])
                        patch_iters.append(self.args.iter_nums)
                        patch_effort.append(0.0)
                    else:
                        _, all_m, n_iters, counts, effort, trace = self.interaction(self.sam, image, label)
                        masks = all_m[-1].unsqueeze(0)
                        patch_iters.append(n_iters)
                        patch_effort.append(effort)
                        for k in all_counts:
                            all_counts[k] += counts.get(k, 0)
                        patch_traces.append(trace)

                    aggregator.add_batch(masks, locations)

                masks_iter_final = aggregator.get_output_tensor()
                dice = self.get_dice_score(torch.sigmoid(masks_iter_final), subject.label.data)
                nsd, hd95 = self._surface_metrics(masks_iter_final, subject.label.data)
                iters_used = int(np.mean(patch_iters))
                n_foreground = sum(1 for e in patch_effort if e > 0)
                total_eff = float(np.mean(patch_effort)) if n_foreground == 0 \
                    else float(np.mean([e for e in patch_effort if e > 0]))
                ces = dice / (iters_used + 1)
                n_fg = max(n_foreground, 1)

                # Aggregate patch traces
                mean_u = float(np.nanmean([
                    t.get('mean_uncert', float('nan')) for t in patch_traces
                ])) if patch_traces else float('nan')
                accept_u = float(np.nanmean([
                    t.get('accept_uncert', float('nan')) for t in patch_traces
                ])) if patch_traces else float('nan')
                early = bool(np.mean([float(t.get('stopped_early', False)) for t in patch_traces]) > 0.5) if patch_traces else False

                sid = self._subject_id(image_path)
                row = {
                    'subject': sid,
                    'dice': float(dice),
                    'nsd': float(nsd),
                    'hd95': float(hd95) if not (isinstance(hd95, float) and math.isnan(hd95)) else '',
                    'iters': int(iters_used),
                    'ces': float(ces),
                    'effort': float(total_eff),
                    'n_point': all_counts['point'] / n_fg,
                    'n_box': all_counts['box'] / n_fg,
                    'n_scribble': all_counts['scribble'] / n_fg,
                    'n_stop': all_counts.get('stop', 0) / n_fg,
                    'stopped_early': int(early),
                    'accept_uncert': accept_u,
                    'mean_uncert': mean_u,
                    'action_seq': '',
                }
                case_rows.append(row)

                dice_summary.append(dice)
                nsd_summary.append(nsd)
                hd95_summary.append(hd95)
                iters_summary.append(iters_used)
                ces_summary.append(ces)
                effort_summary.append(total_eff)
                act_pt_summary.append(all_counts['point'] / n_fg)
                act_bx_summary.append(all_counts['box'] / n_fg)
                act_sc_summary.append(all_counts['scribble'] / n_fg)
                act_st_summary.append(all_counts.get('stop', 0) / n_fg)
                uncert_accept.append(accept_u)
                uncert_mean.append(mean_u)
                early_stop_flags.append(float(early))

                self.logger.info(
                    'case: {}/{} | subject: {}'.format(idx, len(self.val_data), str(image_path)) +
                    ' | dice: {:.4f} nsd: {:.4f} hd95: {} iters: {} ces: {:.4f} effort: {:.2f}'
                    ' [pt={} bx={} sc={} st={}] early={} u_acc={:.2f}'.format(
                        dice, nsd,
                        '{:.2f}'.format(hd95) if not (isinstance(hd95, float) and math.isnan(hd95)) else 'nan',
                        iters_used, ces, total_eff,
                        all_counts['point'], all_counts['box'], all_counts['scribble'],
                        all_counts.get('stop', 0), early, accept_u if not math.isnan(accept_u) else -1))

            self._finalize_report(
                case_rows, dice_summary, nsd_summary, hd95_summary,
                iters_summary, ces_summary, effort_summary,
                act_pt_summary, act_bx_summary, act_sc_summary, act_st_summary,
                dice_at, nsd_at, dice_curve_all,
                uncert_accept, uncert_mean, early_stop_flags,
            )
        return dice_summary

    def validater(self, epoch_num):
        device = self.args.device
        with torch.no_grad():
            case_rows = []
            dice_summary, nsd_summary, hd95_summary = [], [], []
            iters_summary, ces_summary, effort_summary = [], [], []
            act_pt_summary, act_bx_summary, act_sc_summary, act_st_summary = [], [], [], []
            uncert_accept, uncert_mean, early_stop_flags = [], [], []
            dice_at = {k: [] for k in _BUDGET_STEPS}
            nsd_at = {k: [] for k in _BUDGET_STEPS}
            dice_curve_all = []

            for idx, (image, label, image_path, subject_dict_save) in enumerate(self.val_data):
                image, label = image.to(device), label.to(device)
                sid = self._subject_id(image_path)

                if self.args.data == 'kits' and image.size(1) > 1:
                    D2 = int(image.size(2) * 2)
                    final_mask = torch.zeros([1, 1, D2, image.size(3), image.size(4)])
                    label_full = torch.zeros([1, 1, D2, image.size(3), image.size(4)])
                    total_iters, total_effort = 0, 0.0
                    all_counts = {'point': 0, 'box': 0, 'scribble': 0, 'stop': 0}
                    all_masks_ch = []
                    traces = []
                    lbl_ch0 = None
                    for ch in range(image.size(1)):
                        img_ch = image[:, ch, :].unsqueeze(1)
                        lbl_ch = label[:, ch, :].unsqueeze(1)
                        mask_ch, masks_ch, n_iters, counts, effort, trace = self.interaction(
                            self.sam, img_ch, lbl_ch)
                        s, e = ch * image.size(2), (ch + 1) * image.size(2)
                        final_mask[0, 0, s:e] = mask_ch[0, 0].cpu()
                        label_full[0, 0, s:e] = lbl_ch[0, 0].cpu()
                        total_iters += n_iters
                        total_effort += effort
                        for k in all_counts:
                            all_counts[k] += counts.get(k, 0)
                        all_masks_ch.append(masks_ch)
                        traces.append(trace)
                        if ch == 0:
                            lbl_ch0 = lbl_ch.cpu()
                    label = label_full.to(device)
                    iters_used = total_iters // image.size(1)
                    counts = all_counts
                    all_masks = all_masks_ch[0]
                    label_for_budget = lbl_ch0
                    trace = traces[0]
                    final_mask = final_mask.cpu()
                else:
                    final_mask, all_masks, iters_used, counts, total_effort, trace = \
                        self.interaction(self.sam, image, label)
                    final_mask = final_mask.cpu()
                    label_for_budget = label.cpu()

                dice = self.get_dice_score(torch.sigmoid(final_mask), label.cpu())
                nsd, hd95 = self._surface_metrics(final_mask, label.cpu())
                ces = dice / (iters_used + 1)

                # Build / pad dice curve for paper efficiency plots
                dice_curve = list(trace.get('dice_curve', []))
                if not dice_curve and all_masks is not None and all_masks.numel() > 0:
                    for t in range(all_masks.size(0)):
                        dice_curve.append(
                            self.get_dice_score(torch.sigmoid(all_masks[t:t+1]), label_for_budget)
                        )
                # Pad with last dice out to max budget (accepted contour held)
                max_budget = max(self.args.iter_nums, max(_BUDGET_STEPS))
                if dice_curve:
                    while len(dice_curve) < max_budget:
                        dice_curve.append(dice_curve[-1])
                dice_curve_all.append(dice_curve)

                for step in _BUDGET_STEPS:
                    if len(dice_curve) >= step:
                        dice_at[step].append(dice_curve[step - 1])
                    # NSD at fixed budgets from collected masks when available
                    if all_masks is not None and all_masks.size(0) >= step:
                        nsd_s, _ = self._surface_metrics(all_masks[step - 1:step], label_for_budget)
                        nsd_at[step].append(nsd_s)
                    elif dice_curve and len(dice_curve) >= step:
                        # Early-stop: use final mask surface metrics once
                        if step > iters_used:
                            nsd_at[step].append(nsd)

                accept_u = float(trace.get('accept_uncert', float('nan')))
                mean_u = float(trace.get('mean_uncert', float('nan')))
                early = bool(trace.get('stopped_early', False))
                act_seq = ','.join(trace.get('action_seq', []))

                row = {
                    'subject': sid,
                    'dice': float(dice),
                    'nsd': float(nsd),
                    'hd95': float(hd95) if not (isinstance(hd95, float) and math.isnan(hd95)) else '',
                    'iters': int(iters_used),
                    'ces': float(ces),
                    'effort': float(total_effort),
                    'n_point': counts.get('point', 0),
                    'n_box': counts.get('box', 0),
                    'n_scribble': counts.get('scribble', 0),
                    'n_stop': counts.get('stop', 0),
                    'stopped_early': int(early),
                    'hit_max_iters': int(bool(trace.get('hit_max_iters', iters_used >= self.args.iter_nums))),
                    'accept_uncert': accept_u if not math.isnan(accept_u) else '',
                    'mean_uncert': mean_u if not math.isnan(mean_u) else '',
                    'final_prio_vol': trace.get('final_prio_vol', ''),
                    'action_seq': act_seq,
                }
                for step in _BUDGET_STEPS:
                    row[f'dice_at_{step}'] = dice_curve[step - 1] if len(dice_curve) >= step else ''
                case_rows.append(row)

                dice_summary.append(dice)
                nsd_summary.append(nsd)
                hd95_summary.append(hd95)
                iters_summary.append(iters_used)
                ces_summary.append(ces)
                effort_summary.append(total_effort)
                act_pt_summary.append(counts.get('point', 0))
                act_bx_summary.append(counts.get('box', 0))
                act_sc_summary.append(counts.get('scribble', 0))
                act_st_summary.append(counts.get('stop', 0))
                uncert_accept.append(accept_u)
                uncert_mean.append(mean_u)
                early_stop_flags.append(float(early))

                self.logger.info(
                    'case: {}/{} | subject: {}'.format(idx, len(self.val_data), sid) +
                    ' | dice: {:.4f} nsd: {:.4f} hd95: {} iters: {} ces: {:.4f} effort: {:.2f}'
                    ' [pt={} bx={} sc={} st={}] early={} u_acc={} seq=[{}]'.format(
                        dice, nsd,
                        '{:.2f}'.format(hd95) if not (isinstance(hd95, float) and math.isnan(hd95)) else 'nan',
                        iters_used, ces, total_effort,
                        counts.get('point', 0), counts.get('box', 0),
                        counts.get('scribble', 0), counts.get('stop', 0),
                        early,
                        '{:.2f}'.format(accept_u) if not math.isnan(accept_u) else 'nan',
                        act_seq))

            self._finalize_report(
                case_rows, dice_summary, nsd_summary, hd95_summary,
                iters_summary, ces_summary, effort_summary,
                act_pt_summary, act_bx_summary, act_sc_summary, act_st_summary,
                dice_at, nsd_at, dice_curve_all,
                uncert_accept, uncert_mean, early_stop_flags,
            )
        return dice_summary

    def _finalize_report(
        self,
        case_rows,
        dice_summary, nsd_summary, hd95_summary,
        iters_summary, ces_summary, effort_summary,
        act_pt, act_bx, act_sc, act_st,
        dice_at, nsd_at, dice_curve_all,
        uncert_accept, uncert_mean, early_stop_flags,
    ):
        self._log_summary(
            dice_summary, nsd_summary, hd95_summary,
            iters_summary, ces_summary, effort_summary,
            act_pt, act_bx, act_sc, act_st,
            dice_at, nsd_at, dice_curve_all,
            uncert_accept, uncert_mean, early_stop_flags,
        )
        self._write_report_files(
            case_rows, dice_summary, nsd_summary, hd95_summary,
            iters_summary, ces_summary, effort_summary,
            act_pt, act_bx, act_sc, act_st,
            dice_at, nsd_at, dice_curve_all,
            uncert_accept, uncert_mean, early_stop_flags,
        )

    def _log_summary(
        self,
        dice_summary, nsd_summary, hd95_summary,
        iters_summary, ces_summary, effort_summary=None,
        act_pt=None, act_bx=None, act_sc=None, act_st=None,
        dice_at=None, nsd_at=None, dice_curve_all=None,
        uncert_accept=None, uncert_mean=None, early_stop_flags=None,
    ):
        from scipy import stats as sp_stats

        def ci95(data):
            arr = np.asarray(data, dtype=np.float64)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return float('nan'), float('nan'), float('nan')
            mean = arr.mean()
            if len(arr) < 2:
                return mean, mean, mean
            sem = sp_stats.sem(arr)
            t = sp_stats.t.ppf(0.975, len(arr) - 1)
            return mean, mean - sem * t, mean + sem * t

        d_mean, d_lo, d_hi = ci95(dice_summary)
        n_mean, n_lo, n_hi = ci95(nsd_summary)
        h_clean = [x for x in hd95_summary if not (isinstance(x, float) and math.isnan(x))]
        h_mean, h_lo, h_hi = ci95(h_clean) if h_clean else (float('nan'),) * 3
        i_mean, i_lo, i_hi = ci95(iters_summary)
        c_mean, c_lo, c_hi = ci95(ces_summary)
        e_mean = float(np.mean(effort_summary)) if effort_summary else 0.0

        self.logger.info("=" * 72)
        self.logger.info("TEST SUMMARY — primary interactive metrics (agentic stop)")
        self.logger.info("=" * 72)
        self.logger.info(
            "- Accuracy | "
            "Dice: {:.4f} (95% CI [{:.4f}, {:.4f}]) | "
            "NSD@5mm: {:.4f} (95% CI [{:.4f}, {:.4f}]) | "
            "HD95: {:.2f} (95% CI [{:.2f}, {:.2f}])".format(
                d_mean, d_lo, d_hi, n_mean, n_lo, n_hi, h_mean, h_lo, h_hi))
        self.logger.info(
            "- Efficiency | "
            "K̄ (iters): {:.2f} (95% CI [{:.2f}, {:.2f}]) | "
            "Effort: {:.3f} | "
            "CES=d/(K+1): {:.4f} (95% CI [{:.4f}, {:.4f}])".format(
                i_mean, i_lo, i_hi, e_mean, c_mean, c_lo, c_hi))

        # Paper one-liner: Dice% / NSD% / K
        self.logger.info(
            "- Paper table cell | Dice% / NSD% / K̄ = "
            "{:.1f} / {:.1f} / {:.2f}".format(100 * d_mean, 100 * n_mean, i_mean))

        if act_pt is not None:
            mp, mb, ms = np.mean(act_pt), np.mean(act_bx), np.mean(act_sc)
            mstop = np.mean(act_st) if act_st is not None else 0.0
            total_prompt = mp + mb + ms
            total_prompt = max(total_prompt, 1e-8)
            self.logger.info(
                "- Actions (avg / case) | "
                "Point: {:.2f} ({:.0f}%) | Box: {:.2f} ({:.0f}%) | "
                "Scribble: {:.2f} ({:.0f}%) | Stop: {:.2f} | "
                "Pt/Bx/Sc% = {:.0f}/{:.0f}/{:.0f}".format(
                    mp, 100 * mp / total_prompt,
                    mb, 100 * mb / total_prompt,
                    ms, 100 * ms / total_prompt,
                    mstop,
                    100 * mp / total_prompt,
                    100 * mb / total_prompt,
                    100 * ms / total_prompt))

        if early_stop_flags is not None and len(early_stop_flags):
            frac_early = float(np.mean(early_stop_flags))
            self.logger.info(
                "- Stopping | early-stop fraction: {:.1%} | "
                "hit-max-iters: {:.1%}".format(frac_early, 1.0 - frac_early))

        if uncert_accept is not None:
            ua = [u for u in uncert_accept if not (isinstance(u, float) and math.isnan(u))]
            um = [u for u in uncert_mean if not (isinstance(u, float) and math.isnan(u))]
            if ua:
                a_mean, a_lo, a_hi = ci95(ua)
                self.logger.info(
                    "- Predictive uncertainty | "
                    "at accept: {:.3f} (95% CI [{:.3f}, {:.3f}])".format(a_mean, a_lo, a_hi))
            if um:
                m_mean, m_lo, m_hi = ci95(um)
                self.logger.info(
                    "- Predictive uncertainty | "
                    "mean over steps: {:.3f} (95% CI [{:.3f}, {:.3f}])".format(m_mean, m_lo, m_hi))

        if dice_at:
            parts = []
            for step in _BUDGET_STEPS:
                if dice_at.get(step):
                    m, lo, hi = ci95(dice_at[step])
                    parts.append("@{}: {:.4f} [{:.4f},{:.4f}]".format(step, m, lo, hi))
            if parts:
                self.logger.info(
                    "- Fixed-budget Dice (pad last if early stop) | " + " | ".join(parts))

        if nsd_at:
            parts = []
            for step in _BUDGET_STEPS:
                if nsd_at.get(step):
                    m, lo, hi = ci95(nsd_at[step])
                    parts.append("@{}: {:.4f} [{:.4f},{:.4f}]".format(step, m, lo, hi))
            if parts:
                self.logger.info(
                    "- Fixed-budget NSD@5mm | " + " | ".join(parts))

        if dice_curve_all:
            max_t = max(len(c) for c in dice_curve_all)
            curve_means = []
            for t in range(max_t):
                vals = [c[t] for c in dice_curve_all if len(c) > t]
                if vals:
                    curve_means.append(float(np.mean(vals)))
            if curve_means:
                pretty = " → ".join("{:.3f}".format(v) for v in curve_means[: min(11, len(curve_means))])
                self.logger.info("- Mean Dice-vs-step curve | " + pretty)

        self.logger.info(
            "- n_cases={} | agent={} | refine_test={} | max_iters={}".format(
                len(dice_summary),
                'cpc_uga' if getattr(self.args, 'use_cpc_uga', False)
                else ('medsa' if getattr(self.args, 'use_medsa', False) else 'none'),
                bool(getattr(self.args, 'refine_test', False) or getattr(self.args, 'refine', False)),
                self.args.iter_nums,
            ))
        self.logger.info("=" * 72)

    def _write_report_files(
        self,
        case_rows,
        dice_summary, nsd_summary, hd95_summary,
        iters_summary, ces_summary, effort_summary,
        act_pt, act_bx, act_sc, act_st,
        dice_at, nsd_at, dice_curve_all,
        uncert_accept, uncert_mean, early_stop_flags,
    ):
        """Write per-case CSV + aggregate JSON for paper tables/figures."""
        from scipy import stats as sp_stats

        def ci95(data):
            arr = np.asarray(data, dtype=np.float64)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return None
            mean = float(arr.mean())
            if len(arr) < 2:
                return {'mean': mean, 'ci95_lo': mean, 'ci95_hi': mean, 'std': 0.0, 'n': 1}
            sem = float(sp_stats.sem(arr))
            t = float(sp_stats.t.ppf(0.975, len(arr) - 1))
            return {
                'mean': mean,
                'ci95_lo': mean - sem * t,
                'ci95_hi': mean + sem * t,
                'std': float(arr.std(ddof=1)),
                'n': int(len(arr)),
            }

        stamp = time.strftime('%Y%m%d-%H%M%S')
        tag = getattr(self.args, 'save_name', 'test')
        csv_path = os.path.join(self.report_dir, f'{tag}_per_case_{stamp}.csv')
        json_path = os.path.join(self.report_dir, f'{tag}_summary_{stamp}.json')
        curve_path = os.path.join(self.report_dir, f'{tag}_dice_curve_{stamp}.csv')

        if case_rows:
            fieldnames = list(case_rows[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(case_rows)
            self.logger.info(f"- Wrote per-case CSV: {csv_path}")

        # Dice-vs-step curve CSV (for plotting)
        if dice_curve_all:
            max_t = max(len(c) for c in dice_curve_all)
            with open(curve_path, 'w', newline='') as f:
                w = csv.writer(f)
                header = ['subject'] + [f'step_{t+1}' for t in range(max_t)]
                w.writerow(header)
                for i, curve in enumerate(dice_curve_all):
                    sid = case_rows[i]['subject'] if i < len(case_rows) else f'case_{i}'
                    row = [sid] + list(curve) + [''] * (max_t - len(curve))
                    w.writerow(row[: 1 + max_t])
                # mean row
                means = []
                for t in range(max_t):
                    vals = [c[t] for c in dice_curve_all if len(c) > t]
                    means.append(float(np.mean(vals)) if vals else '')
                w.writerow(['MEAN'] + means)
            self.logger.info(f"- Wrote Dice-vs-step CSV: {curve_path}")

        mp = float(np.mean(act_pt)) if act_pt else 0.0
        mb = float(np.mean(act_bx)) if act_bx else 0.0
        ms = float(np.mean(act_sc)) if act_sc else 0.0
        mstop = float(np.mean(act_st)) if act_st else 0.0
        tot = max(mp + mb + ms, 1e-8)

        summary = {
            'dataset': getattr(self.args, 'data', ''),
            'save_name': tag,
            'agent': (
                'cpc_uga' if getattr(self.args, 'use_cpc_uga', False)
                else ('medsa' if getattr(self.args, 'use_medsa', False) else 'none')
            ),
            'n_cases': len(dice_summary),
            'max_iters': int(self.args.iter_nums),
            'uncert_low': float(getattr(self.args, 'uncert_low', 0.35)),
            'uncert_high': float(getattr(self.args, 'uncert_high', 0.65)),
            'metrics': {
                'dice': ci95(dice_summary),
                'nsd_5mm': ci95(nsd_summary),
                'hd95': ci95([x for x in hd95_summary if not (isinstance(x, float) and math.isnan(x))]),
                'iters_K': ci95(iters_summary),
                'effort': ci95(effort_summary) if effort_summary else None,
                'ces': ci95(ces_summary),
            },
            'paper_cell': {
                'dice_pct': 100 * float(np.mean(dice_summary)) if dice_summary else None,
                'nsd_pct': 100 * float(np.mean(nsd_summary)) if nsd_summary else None,
                'K_bar': float(np.mean(iters_summary)) if iters_summary else None,
            },
            'actions': {
                'point_mean': mp,
                'box_mean': mb,
                'scribble_mean': ms,
                'stop_mean': mstop,
                'point_pct': 100 * mp / tot,
                'box_pct': 100 * mb / tot,
                'scribble_pct': 100 * ms / tot,
            },
            'stopping': {
                'early_stop_fraction': float(np.mean(early_stop_flags)) if early_stop_flags else None,
            },
            'uncertainty': {
                'accept': ci95([u for u in (uncert_accept or []) if not (isinstance(u, float) and math.isnan(u))]),
                'mean_over_steps': ci95([u for u in (uncert_mean or []) if not (isinstance(u, float) and math.isnan(u))]),
            },
            'fixed_budget_dice': {
                str(k): ci95(v) for k, v in (dice_at or {}).items() if v
            },
            'fixed_budget_nsd': {
                str(k): ci95(v) for k, v in (nsd_at or {}).items() if v
            },
            'files': {
                'per_case_csv': csv_path if case_rows else None,
                'dice_curve_csv': curve_path if dice_curve_all else None,
            },
        }
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"- Wrote summary JSON: {json_path}")

    def get_next_click3D_torch_2(self, prev_seg, gt_semantic_seg):

        mask_threshold = 0.5

        batch_points = []
        batch_labels = []
        # dice_list = []

        pred_masks = (prev_seg > mask_threshold)
        true_masks = (gt_semantic_seg > 0)
        fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
        fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)
        print('fn: {}, fp: {}'.format(torch.count_nonzero(fn_masks) / torch.count_nonzero(true_masks),
                                      torch.count_nonzero(fp_masks) / torch.count_nonzero(true_masks)))
        to_point_mask = torch.logical_or(fn_masks, fp_masks)
        #to_point_mask = fn_masks
        for i in range(gt_semantic_seg.shape[0]):
            bp_list, bl_list = [], []
            points = torch.argwhere(to_point_mask[i])
            if self.args.num_clicks > len(points):
                click_size = len(points)
            else:
                click_size = self.args.num_clicks

            dynamic_size = random.randint(1, click_size) if self.args.dynamic else click_size

            point_index = np.random.choice(len(points), size=dynamic_size, replace=False)
            points_select = points[point_index]  # each row tensor([0, x, y, z]), size --> num_clicks x 4
            # point = points[np.random.randint(len(points))] # tensor([0, x, y, z])
            for click_index in range(dynamic_size):
                point = points_select[click_index]
                if fn_masks[i, 0, point[1], point[2], point[3]]:
                    is_positive = True
                else:
                    is_positive = False

                bp = point[1:].clone().detach().reshape(1, 1, 3)
                bl = torch.tensor([int(is_positive), ]).reshape(1, 1)
                bp_list.append(bp)
                bl_list.append(bl)

            if self.args.use_scribble:
                #sample_method = random.choice(['line', 'center', 'default'])
                sample_method = 'center'
                scribble_types = {
                    'line': 'LineScribble',
                    'center': 'CenterlineScribble',
                    'default': 'ContourScribble'
                }

                def create_scribble_mask(scribble_type, data):
                    scribble_object = getattr(scribble, scribble_type)()
                    scribble_mask = scribble_object.batch_scribble(data).permute(1, 2, 3, 0)
                    return scribble_mask > 0

                # fg = gt_semantic_seg[i].permute(3, 0, 1, 2).float()
                # bg = (torch.ones_like(pred_masks[i, :]).float() - gt_semantic_seg[i].float()).permute(3, 0, 1, 2)
                fg, bg = fn_masks[0].permute(3, 0, 1, 2).float(), fp_masks[0].permute(3, 0, 1, 2).float()

                scribble_type = scribble_types.get(sample_method, scribble_types['default'])

                scribble_mask_fg = create_scribble_mask(scribble_type, fg)
                #fg_coors = torch.argwhere(scribble_mask_fg)[:, 1:].unsqueeze(0)[:, 0: 100, :]  # for computation only
                fg_coors = torch.argwhere(scribble_mask_fg)[:, 1:].unsqueeze(0)
                if self.args.efficient_scribble:
                    fg_coors = fg_coors[:, 0: 10000, :]  # for computation only# for computation only
                fg_coors_label = torch.ones(1, fg_coors.size(1))
                bp_list.append(fg_coors)
                bl_list.append(fg_coors_label)
                # x,y,z = bp_list[-1][0, 99, 0], bp_list[-1][0, 99, 1], bp_list[-1][0, 99, 2]
                # print(gt_semantic_seg[i, 0, x,y,z])

                #if sample_method == 'default':
                if torch.count_nonzero(fp_masks) > 0:
                    scribble_mask_bg = create_scribble_mask(scribble_type, bg)
                    bg_coors = torch.argwhere(scribble_mask_bg)[:, 1:].unsqueeze(0)
                    if self.args.efficient_scribble:
                        bg_coors = bg_coors[:, 0: 10000, :]
                    bg_coors_label = torch.zeros(1, bg_coors.size(1))
                    bp_list.append(bg_coors)
                    bl_list.append(bg_coors_label)

            batch_points.append(torch.cat(bp_list, dim=1))
            batch_labels.append(torch.cat(bl_list, dim=1))

            smallest_n = min(tensor.size(1) for tensor in batch_labels)
            batch_points = [tensor[:, :smallest_n] if tensor.size(1) > smallest_n else tensor for tensor in
                            batch_points]
            batch_labels = [tensor[:, :smallest_n] if tensor.size(1) > smallest_n else tensor for tensor in
                            batch_labels]

            # Check the shapes of the adjusted tensors
            for i, tensor in enumerate(batch_points):
                print(f"Tensor {i + 1} shape: {tensor.shape}")


        return batch_points, batch_labels

    def get_points(self, prev_masks, label, action=ACTION_POINT, priority_mask=None):
        """
        Action-conditioned prompt sampler (mirrors trainer._get_next_point_medsa).
          ACTION_POINT   → clicks sampled from FN∪FP (or clinical-priority CC)
          ACTION_BOX     → bounding box of GT + clicks
          ACTION_SCRIBBLE → CenterlineScribble skeleton of top-10% FN∪FP voxels
        """
        dev = self.args.device
        num_clicks = self.args.num_clicks_validation

        batch_points, batch_labels = [], []
        pred_masks = (prev_masks > 0.5)
        true_masks = (label > 0)
        fn_masks   = (true_masks & ~pred_masks)
        fp_masks   = (~true_masks & pred_masks)

        use_prio = (
            (
                getattr(self.args, 'use_clinical_priority', False)
                or getattr(self.args, 'use_cpc_uga', False)
            )
            and priority_mask is not None
            and priority_mask.sum() > 0
        )

        for i in range(label.shape[0]):
            bp_list, bl_list = [], []

            if use_prio:
                prio = priority_mask
                while prio.dim() < fn_masks[i].dim():
                    prio = prio.unsqueeze(0)
                prio = (prio > 0.5).to(fn_masks[i].device)
                fn_i = fn_masks[i] & prio
                fp_i = fp_masks[i] & prio
                if (fn_i | fp_i).sum() == 0:
                    fn_i, fp_i = fn_masks[i], fp_masks[i]
            else:
                fn_i, fp_i = fn_masks[i], fp_masks[i]

            if action == ACTION_SCRIBBLE and getattr(self.args, 'use_scribble', False):
                error_vol  = (fn_i.float() + fp_i.float()).clamp(0, 1)
                k_top      = max(1, int(error_vol.numel() * 0.10))
                threshold  = error_vol.flatten().topk(k=k_top).values.min()
                scribble_obj = scribble.CenterlineScribble()
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
                        coords = coords[:, :min(coords.size(1), 500), :]
                        bl_list.append(torch.full((1, coords.size(1)), float(polarity)))
                        bp_list.append(coords)

            # Fallback / POINT action: clicks from FN∪FP (priority-restricted)
            if not bp_list:
                to_point = torch.logical_or(fn_i, fp_i)
                points   = torch.argwhere(to_point)
                if len(points) == 0:
                    points = torch.argwhere(true_masks[i])
                if len(points) == 0:
                    points = label.new_zeros(1, 4, dtype=torch.long)
                click_size = max(min(num_clicks, len(points)), 1)
                selected   = points[np.random.choice(len(points), click_size, replace=False)]
                for ci in range(len(selected)):
                    pt     = selected[ci]
                    is_pos = bool(fn_masks[i, 0, pt[1], pt[2], pt[3]])
                    bp_list.append(pt[1:].clone().detach().reshape(1, 1, 3))
                    bl_list.append(torch.tensor([float(int(is_pos))]).reshape(1, 1))

            batch_points.append(torch.cat(bp_list, dim=1))
            batch_labels.append(torch.cat(bl_list, dim=1))

        # Pad to same length across batch
        smallest_n   = min(t.size(1) for t in batch_labels)
        batch_points = [t[:, :smallest_n] for t in batch_points]
        batch_labels = [t[:, :smallest_n] for t in batch_labels]

        points_co = torch.cat(batch_points, dim=0).to(dev)
        points_la = torch.cat(batch_labels, dim=0).to(dev)
        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        # Bounding box: only for BOX action (or always-on when agent disabled)
        agent_on = self.use_medsa or self.use_cpc_uga
        if action == ACTION_BOX or not agent_on:
            bbox_coords = _bbox_mask(label[:, 0, :]).to(dev) if self.args.use_box else None
        elif action != ACTION_SCRIBBLE and self.args.use_box:
            # Supply box as secondary cue for POINT action too
            bbox_coords = _bbox_mask(label[:, 0, :]).to(dev)
        else:
            bbox_coords = None

        return points_co, points_la, bbox_coords


    def batch_forward(self, sam_model, features, image_embedding, image, prev_masks,
                      points=None, boxes=None, medsa_context=None):
        prev_masks = F.interpolate(prev_masks, scale_factor=0.25)
        features = [features[i].to(self.args.device) for i in range(0, len(features))]

        new_point_embedding, new_image_embedding = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=prev_masks,
            image_embeddings=image_embedding.to(self.args.device)
        )

        mask, pred_dice, spatial_hmap, _ = sam_model.mask_decoder(
            prompt_embeddings=new_point_embedding,
            image_embeddings=new_image_embedding,
            feature_list=features,
            medsa_context=medsa_context,
        )

        return mask, pred_dice, spatial_hmap

    # Effort cost per action type (matches reward definition in policy.py)
    _EFFORT = {ACTION_POINT: 0.1, ACTION_BOX: 0.3, ACTION_SCRIBBLE: 0.5, ACTION_STOP: 0.0}

    def _use_refine(self) -> bool:
        """Match training validation: refine whenever --refine is set (or legacy --refine_test)."""
        return bool(getattr(self.args, 'refine', False) or getattr(self.args, 'refine_test', False))

    def _interaction_cpc_uga(self, sam_model, image, label):
        """CPC–UGA / dual-uncertainty test loop (no CoCC). Returns rich per-case trace."""
        max_iters = self.args.iter_nums
        _fk = getattr(self.args, 'fixed_k', -1)
        if getattr(self.args, 'force_action', None) and _fk and _fk > 0:
            max_iters = min(max_iters, _fk)

        image_embedding, feature_list = sam_model.image_encoder(image)
        self.click_points = []
        self.click_labels = []
        prev_masks = torch.zeros_like(label).to(label.device)
        collected = []
        iters_used = 0
        action_counts = {
            ACTION_POINT: 0, ACTION_BOX: 0, ACTION_SCRIBBLE: 0, ACTION_STOP: 0,
        }
        total_effort = 0.0
        prev_dice = 0.0
        prev_uncert = 1.0

        dice_curve, uncert_curve, prio_vol_curve, action_seq = [], [], [], []
        stopped_early = False
        accept_uncert = float('nan')
        last_prio_vol = float('nan')

        _force = getattr(self.args, 'force_action', None)
        _force_map = {'point': ACTION_POINT, 'box': ACTION_BOX, 'scribble': ACTION_SCRIBBLE}
        forced_action = _force_map.get(_force) if _force else None

        for iter_num in range(max_iters):
            prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks
            action = ACTION_POINT
            priority_mask = None

            label_5d = _ensure_5d(label)
            pred_5d = _ensure_5d(prev_masks_sigmoid)
            image_5d = _ensure_5d(image)
            pred_bin = (pred_5d > 0.5)
            true_bin = (label_5d > 0)
            fn_mask = (true_bin & ~pred_bin).float()
            fp_mask = (~true_bin & pred_bin).float()

            with torch.no_grad():
                geo = compute_geometric_descriptors(fn_mask, fp_mask, pred_5d)
                priority_mask, _, _ = priority_mask_from_errors(
                    fn_mask[0], fp_mask[0], pred_5d[0], image=image_5d[0],
                    lam_iso=getattr(self.args, 'prio_lam_iso', 1.0),
                    lam_int=getattr(self.args, 'prio_lam_int', 0.5),
                    lam_bnd=getattr(self.args, 'prio_lam_bnd', 0.5),
                )
                prio_vol = float(priority_mask.sum().item()) / (float(fn_mask[0].numel()) + 1e-8)
                last_prio_vol = prio_vol
                bnd = float(geo[0, 3].item())

                if forced_action is not None:
                    action = forced_action
                elif iter_num > 0:
                    curr_dice = self.get_dice_score(prev_masks_sigmoid, label)
                    state = PriorityUncertaintyPolicy.build_state(
                        dice_current=float(curr_dice),
                        delta_dice=float(curr_dice) - prev_dice,
                        iter_progress=iter_num / max(max_iters, 1),
                        priority_volume=prio_vol,
                        uncertainty=prev_uncert,
                        device=self.args.device,
                    )
                    action = self.type_policy.select_action(
                        state, epsilon=0.0,
                        priority_volume=prio_vol,
                        boundary_overlap=bnd,
                        iter_idx=iter_num,
                        max_iters=max_iters,
                        dice_current=float(curr_dice),
                        uncertainty=prev_uncert,
                        uncert_low=getattr(self.args, 'uncert_low', 0.35),
                        uncert_high=getattr(self.args, 'uncert_high', 0.65),
                    )
                    prev_dice = float(curr_dice)
                    if action == ACTION_STOP:
                        action_counts[ACTION_STOP] += 1
                        action_seq.append('stop')
                        stopped_early = True
                        accept_uncert = float(prev_uncert)
                        break

            iters_used += 1
            action_counts[action] = action_counts.get(action, 0) + 1
            total_effort += self._EFFORT.get(action, 0.1)
            action_seq.append(_ACT_NAME.get(action, '?'))
            prio_vol_curve.append(prio_vol)

            points_input, labels_input, bbox_input = self.get_points(
                prev_masks_sigmoid, label, action=action, priority_mask=priority_mask,
            )
            mask, pred_dice_score, _ = self.batch_forward(
                sam_model, feature_list, image_embedding, image, prev_masks,
                points=[points_input, labels_input], boxes=bbox_input,
                medsa_context=None,
            )
            prev_uncert = mask_uncertainty(mask, pred_dice_score)
            uncert_curve.append(float(prev_uncert))

            B = mask.size(0)
            if self.args.multiple_outputs:
                _, max_idx = torch.max(pred_dice_score, dim=1)
                b_idx = torch.arange(B, device=mask.device)
                mask_best = mask[b_idx, max_idx].unsqueeze(1)
            else:
                mask_best = mask

            if self._use_refine():
                mask_refine, _ = self.sam.mask_decoder.refine(
                    image, mask_best, [self.click_points, self.click_labels], mask_best.detach())
                mask_best = mask_refine

            prev_masks = mask_best
            collected.append(prev_masks.detach().cpu())
            dice = self.get_dice_score(torch.sigmoid(prev_masks).cpu().numpy(), label.cpu().numpy())
            dice_curve.append(float(dice))
            act_name = _ACT_NAME.get(action, '?')
            print(
                f'iter: {iters_used} | action: {act_name} | u={prev_uncert:.2f} '
                f'| prio_vol={prio_vol:.4g} | Dice: {dice:.4f}'
            )

        all_masks = torch.cat(collected, dim=0) if collected else torch.zeros(
            [1, 1, image.size(2), image.size(3), image.size(4)])
        counts = {
            'point': action_counts.get(ACTION_POINT, 0),
            'box': action_counts.get(ACTION_BOX, 0),
            'scribble': action_counts.get(ACTION_SCRIBBLE, 0),
            'stop': action_counts.get(ACTION_STOP, 0),
        }
        if not stopped_early and iters_used >= max_iters:
            accept_uncert = float(prev_uncert)
        mean_u = float(np.mean(uncert_curve)) if uncert_curve else float('nan')
        trace = {
            'dice_curve': dice_curve,
            'uncert_curve': uncert_curve,
            'prio_vol_curve': prio_vol_curve,
            'action_seq': action_seq,
            'stopped_early': stopped_early,
            'hit_max_iters': (not stopped_early) and iters_used >= max_iters,
            'accept_uncert': accept_uncert,
            'mean_uncert': mean_u,
            'final_prio_vol': last_prio_vol,
            'n_stop': counts['stop'],
        }
        return prev_masks, all_masks, iters_used, counts, total_effort, trace

    def interaction(self, sam_model, image, label):
        """
        Run the interaction loop with optional MedSA / CPC–UGA policy-driven stopping.
        Returns: (final_mask, all_masks, iters_used, action_counts, total_effort, trace)
        """
        if self.use_cpc_uga:
            return self._interaction_cpc_uga(sam_model, image, label)

        max_iters = self.args.iter_nums
        # When running a forced-action ablation, cap to --fixed_k if specified
        _fk = getattr(self.args, 'fixed_k', -1)
        if getattr(self.args, 'force_action', None) and _fk and _fk > 0:
            max_iters = min(max_iters, _fk)

        image_embedding, feature_list = sam_model.image_encoder(image)

        self.click_points = []
        self.click_labels = []
        prev_masks    = torch.zeros_like(label).to(label.device)
        collected     = []
        iters_used    = 0
        action_counts = {
            ACTION_POINT: 0, ACTION_BOX: 0, ACTION_SCRIBBLE: 0, ACTION_STOP: 0,
        }
        total_effort  = 0.0
        dice_curve, action_seq = [], []
        stopped_early = False

        # Resolve forced action once (None means let the policy decide)
        _force = getattr(self.args, 'force_action', None)
        _force_map = {'point': ACTION_POINT, 'box': ACTION_BOX, 'scribble': ACTION_SCRIBBLE}
        forced_action = _force_map.get(_force) if _force else None

        # MedSA session state
        if self.use_medsa:
            self.memory.reset()
            e_full            = torch.zeros(image.size(0), 132, device=self.args.device)
            m_i               = torch.zeros(image.size(0), 132, device=self.args.device)
            prev_dice         = 0.0
            prev_spatial_hmap = None
            f_persistent      = False
            persist_region    = None

        for iter_num in range(max_iters):
            prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

            # ── MedSA: encode error, query policy ────────────────────────
            action = ACTION_POINT   # default (also used when use_medsa=False)
            medsa_context = None
            priority_mask = None
            if forced_action is not None:
                action = forced_action   # bypass policy entirely
            if self.use_medsa:
                label_5d = _ensure_5d(label)
                pred_5d  = _ensure_5d(prev_masks_sigmoid)
                image_5d = _ensure_5d(image)
                pred_bin = (pred_5d > 0.5)
                true_bin = (label_5d > 0)
                fn_mask  = (true_bin & ~pred_bin).float()
                fp_mask  = (~true_bin & pred_bin).float()

                with torch.no_grad():
                    geo    = compute_geometric_descriptors(fn_mask, fp_mask, pred_5d)
                    e_full = self.edit_encoder(fn_mask, fp_mask, pred_bin.float(), image_5d, geo)
                    m_i, f_persistent = self.memory.update(e_full)
                    persist_region = ((fn_mask + fp_mask) > 0).float() if f_persistent else None

                    medsa_context = {
                        'e_full': e_full.detach(),
                        'm_i': m_i.detach(),
                        'f_persistent': f_persistent,
                        'persist_region': persist_region,
                    }

                    if getattr(self.args, 'use_clinical_priority', False):
                        priority_mask, _, _ = priority_mask_from_errors(
                            fn_mask[0], fp_mask[0], pred_5d[0], image=image_5d[0],
                            lam_iso=getattr(self.args, 'prio_lam_iso', 1.0),
                            lam_int=getattr(self.args, 'prio_lam_int', 0.5),
                            lam_bnd=getattr(self.args, 'prio_lam_bnd', 0.5),
                        )

                    if forced_action is None and iter_num > 0:
                        e_cnn       = e_full[:, :128]
                        m_cnn       = m_i[:, :128]
                        e_norm      = torch.norm(e_cnn, dim=1).mean().item()
                        e_mag_norm  = e_norm / (e_norm + 1.0)
                        persistence = float(F.cosine_similarity(e_cnn, m_cnn, dim=1).mean().item())

                        if prev_spatial_hmap is not None:
                            h_flat = torch.sigmoid(prev_spatial_hmap).flatten().float()
                            h_flat = h_flat / (h_flat.sum() + 1e-8)
                            raw_ent = -(h_flat * torch.log(h_flat + 1e-10)).sum().item()
                            spatial_entropy = raw_ent / math.log(float(h_flat.numel()) + 1.0)
                        else:
                            spatial_entropy = 1.0

                        curr_dice = self.get_dice_score(prev_masks_sigmoid, label)
                        state = EditConditionedTypePolicy.build_state(
                            dice_current=float(curr_dice),
                            delta_dice=float(curr_dice) - prev_dice,
                            iter_progress=iter_num / max(max_iters, 1),
                            edit_volume=float(geo[0, 0].item()),
                            edit_bnd_ovlp=float(geo[0, 3].item()),
                            error_magnitude=e_mag_norm,
                            persistence=persistence,
                            spatial_entropy=spatial_entropy,
                            device=self.args.device,
                        )
                        action = self.type_policy.select_action(
                            state, epsilon=0.0,
                            edit_volume=float(geo[0, 0].item()),
                            iter_idx=iter_num,
                            max_iters=max_iters,
                            dice_current=float(curr_dice),
                            spatial_entropy=spatial_entropy,
                            persistence=persistence,
                            use_uncertainty_gate=getattr(self.args, 'use_uncertainty_gate', False),
                            uncert_low=getattr(self.args, 'uncert_low', 0.35),
                            uncert_high=getattr(self.args, 'uncert_high', 0.65),
                            persist_thresh=getattr(self.args, 'persist_thresh', 0.85),
                        )
                        prev_dice = float(curr_dice)

                        if action == ACTION_STOP:
                            action_counts[ACTION_STOP] += 1
                            action_seq.append('stop')
                            stopped_early = True
                            break

            iters_used += 1
            action_counts[action] = action_counts.get(action, 0) + 1
            total_effort += self._EFFORT.get(action, 0.1)
            action_seq.append(_ACT_NAME.get(action, '?'))

            # ── Action-conditioned prompts ────────────────────────────────
            points_input, labels_input, bbox_input = self.get_points(
                prev_masks_sigmoid, label, action=action, priority_mask=priority_mask,
            )

            mask, pred_dice_score, spatial_hmap = self.batch_forward(
                sam_model, feature_list, image_embedding, image, prev_masks,
                points=[points_input, labels_input], boxes=bbox_input,
                medsa_context=medsa_context,
            )
            if self.use_medsa and spatial_hmap is not None:
                prev_spatial_hmap = spatial_hmap.detach()

            B = mask.size(0)
            if self.args.multiple_outputs:
                _, max_idx = torch.max(pred_dice_score, dim=1)
                b_idx      = torch.arange(B, device=mask.device)
                mask_best  = mask[b_idx, max_idx].unsqueeze(1)
            else:
                mask_best = mask

            if self._use_refine():
                mask_refine, _ = self.sam.mask_decoder.refine(
                    image, mask_best, [self.click_points, self.click_labels], mask_best.detach())
                mask_best = mask_refine

            prev_masks = mask_best
            collected.append(prev_masks.detach().cpu())

            dice = self.get_dice_score(torch.sigmoid(prev_masks).cpu().numpy(), label.cpu().numpy())
            dice_curve.append(float(dice))
            act_name = _ACT_NAME.get(action, '?')
            print(f'iter: {iters_used} | action: {act_name} | Dice: {dice:.4f}')

        all_masks = torch.cat(collected, dim=0) if collected else torch.zeros(
            [1, 1, image.size(2), image.size(3), image.size(4)])

        counts = {
            'point':   action_counts.get(ACTION_POINT,    0),
            'box':     action_counts.get(ACTION_BOX,      0),
            'scribble':action_counts.get(ACTION_SCRIBBLE, 0),
            'stop':    action_counts.get(ACTION_STOP,     0),
        }
        trace = {
            'dice_curve': dice_curve,
            'uncert_curve': [],
            'prio_vol_curve': [],
            'action_seq': action_seq,
            'stopped_early': stopped_early,
            'hit_max_iters': (not stopped_early) and iters_used >= max_iters,
            'accept_uncert': float('nan'),
            'mean_uncert': float('nan'),
            'final_prio_vol': float('nan'),
            'n_stop': counts['stop'],
        }
        return prev_masks, all_masks, iters_used, counts, total_effort, trace



    def _interaction(self, sam_model, image, label, iter_nums, train=False, return_each_iter=False):
        if return_each_iter:
            return_mask_total_iter = torch.zeros([iter_nums, 1, image.size(2), image.size(3), image.size(4)])

        image_embedding, feature_list = self.sam.image_encoder(image)
        self.click_points = []
        self.click_labels = []
        return_loss  = 0
        actual_iters = 0
        prev_masks   = torch.zeros_like(label, dtype=torch.float).to(label.device)

        for iter_num in range(iter_nums):
            actual_iters += 1
            prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

            if self.args.init_learning and iter_num == 0:
                boundary, margin, content = boundary_selection.find_boundary_map(label)
                use_content = True
                for batch_index in range(label.size(0)):
                    if torch.count_nonzero(content[batch_index]) < self.args.num_clicks:
                        use_content = False
                label_sample = content if use_content else label
            else:
                label_sample = label

            points_input, labels_input, box_input = self.get_points(prev_masks_sigmoid, label_sample)
            mask, dice_pred, _ = self.batch_forward(
                sam_model, feature_list, image_embedding, image, prev_masks,
                points=[points_input, labels_input], boxes=box_input
            )

            # ── Candidate selection (IoU-based) ───────────────────────────
            B = mask.size(0)
            if self.args.multiple_outputs:
                _, max_label_index = torch.max(dice_pred, dim=1)
                b_idx              = torch.arange(B, device=mask.device)
                mask_best          = mask[b_idx, max_label_index].unsqueeze(1)
            else:
                mask_best = mask

            if self.args.refine and self.args.refine_test:
                mask_refine, _ = self.sam.mask_decoder.refine(
                    image, mask_best, [self.click_points, self.click_labels], mask_best.detach())
                self.logger.info(
                    'dice before refine {} and after {}, label 0: {}, label 1: {}'.format(
                        self.get_dice_score(torch.sigmoid(mask_best), label),
                        self.get_dice_score(torch.sigmoid(mask_refine), label),
                        str(labels_input.numel() - torch.count_nonzero(labels_input)),
                        str(torch.count_nonzero(labels_input))))
                mask_best = mask_refine

            loss        = self.get_dice_score(torch.sigmoid(mask_best), label)
            return_loss += loss
            prev_masks   = mask_best

            if return_each_iter:
                return_mask_total_iter[iter_num, :] = mask_best

        if return_each_iter:
            print(return_mask_total_iter.shape)
            return return_loss / actual_iters, return_mask_total_iter
        else:
            return return_loss / actual_iters, prev_masks

    def get_dice_score(self, prev_masks, label):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.nan
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        if torch.is_tensor(prev_masks):
            pred_masks = (prev_masks > 0.5).to(prev_masks.device)
            true_masks = (label > 0).to(prev_masks.device)
        else:
            pred_masks = (prev_masks > 0.5)
            true_masks = (label > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        mean = sum(dice_list) / len(dice_list)
        return float(mean.item() if torch.is_tensor(mean) else mean)




def main():
    init_seeds()
    args = parser.parse_args()
    check_and_setup_parser(args)

    log_name = 'test_' + args.save_name
    setup_logger(logger_name=log_name, root=args.save_dir, screen=True, tofile=True)
    logger = logging.getLogger(log_name)
    logger.info(str(args))

    #ckpt = '/home/hao/Hao/3D_medical_foundation_model/src/implementation/log/colon/3DSAM/best.pth.tar'
    ckpt = os.path.join(args.save_dir, args.checkpoint + '.pth.tar')
    with torch.no_grad():
        tester = Tester(args, logger, ckpt)
        loss = tester.validate(epoch_num=0)

        print(loss)

    logger.info("- Test done")

if __name__ == "__main__":
    main()