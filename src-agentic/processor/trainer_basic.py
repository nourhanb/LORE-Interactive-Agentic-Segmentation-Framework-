from abc import abstractmethod
import torch
import numpy as np
from torch.optim import AdamW, lr_scheduler
from config.config_setup import build_model, get_dataloader
from config.config_args import use_sliding_window
from monai.losses import DiceCELoss, DiceLoss
import torch.nn as nn
from utils.util import save_checkpoint
import time
import os
import torch.distributed as dist
from torch.cuda import amp
import torchio as tio


def _tio_volume_4d(t):
    """Ensure a torchio-ready volume tensor shaped [C, D, H, W]."""
    t = t.float() if isinstance(t, torch.Tensor) else torch.as_tensor(t).float()
    while t.dim() > 4:
        t = t.squeeze(0)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() != 4:
        raise ValueError(f'Expected 4D [C,D,H,W] for torchio, got shape {tuple(t.shape)}')
    return t


def _subject_from_dict(subject_dict):
    """Build a torchio Subject from the LiTS / full-FOV dict batch format."""
    img = _tio_volume_4d(subject_dict['image']['data'][0])
    lab = _tio_volume_4d(subject_dict['label']['data'][0])
    img_aff = subject_dict['image']['affine'][0]
    lab_aff = subject_dict['label']['affine'][0]
    if isinstance(img_aff, torch.Tensor):
        while img_aff.dim() > 2:
            img_aff = img_aff.squeeze(0)
    if isinstance(lab_aff, torch.Tensor):
        while lab_aff.dim() > 2:
            lab_aff = lab_aff.squeeze(0)
    return tio.Subject(
        image=tio.ScalarImage(tensor=img, affine=img_aff),
        label=tio.LabelMap(tensor=lab, affine=lab_aff),
    )


def _pad_subject_for_grid(subject, patch_size=128):
    """Pad any axis shorter than patch_size so GridSampler can run."""
    spatial = tuple(int(s) for s in subject.spatial_shape)
    target = tuple(max(s, int(patch_size)) for s in spatial)
    if spatial != target:
        subject = tio.CropOrPad(target_shape=target)(subject)
    return subject


class Trainer_basic(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        a = time.time()
        use_small = True if self.args.use_small_dataset else False
        self.train_data, self.val_data = get_dataloader(args, split='train', use_small=use_small), get_dataloader(args, split='val', use_small=use_small)
        if self.args.use_sam3d_turbo:
            self.sam = build_model(args, checkpoint='../src/ckpt/sam_med3d_turbo.pth')
        else:
            self.sam = build_model(args)
        if self.args.ddp:
            self.sam = self.sam.module

        self.best_dice, self.best_epoch, self.start_epoch = 0, 0, 0
        self.pooling_layer = nn.AvgPool3d((self.args.boundary_kernel_size, self.args.boundary_kernel_size, 1), stride=1,
                                     padding=(int((self.args.boundary_kernel_size - 1) / 2),
                                              int((self.args.boundary_kernel_size - 1) / 2),
                                              0)).cuda()

        self.setup()
        print('dataloaders are created, models are loaded, and others are set, spent {} for rank {}'
              .format(round(time.time() - a, 2), self.args.rank))


    def run(self):
        self.scaler = amp.GradScaler()
        for epoch_num in range(self.start_epoch, self.args.max_epoch):
            self.sam.train()
            if self.args.ddp:
                # dist.barrier() # set a barrier until all processes are at same point
                self.train_data.sampler.set_epoch(epoch_num)

            self.train(epoch_num)
            if self.args.ddp and self.args.rank == 0:
                print('doing validation on rank=0')
                current_mean_dice = (
                    self.validate_sliding_window(epoch_num)
                    if use_sliding_window(self.args) else self.validate(epoch_num)
                )
            else:
                current_mean_dice = (
                    self.validate_sliding_window(epoch_num)
                    if use_sliding_window(self.args) else self.validate(epoch_num)
                )
            # https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
            # if self.args.ddp:
                # dist.barrier()
            self.save_model(current_mean_dice, epoch_num)

    @abstractmethod
    def forward(self, model, image, label, iter_nums, train, return_each_iter=False, epoch=0):
        pass

    @staticmethod
    def _fmt(val):
        """Format a loss float for logging — always 4 decimal places."""
        return f"{val:.4f}"

    def train(self, epoch_num):
        loss_summary = []
        for idx, (image, label, image_path) in enumerate(self.train_data):
            self.optimizer.zero_grad()

            image, label = image.to(self.args.device), label.to(self.args.device)
            with amp.autocast():
                loss, _ = self.forward(self.sam, image, label,
                                       iter_nums=self.args.iter_nums, train=True,
                                       epoch=epoch_num)

            loss_summary.append(loss.detach().cpu().numpy())

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.sam.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ── DQN update (outside amp.autocast to preserve grad_fn) ────
            # forward() populates the replay buffer and stores _last_lam_rl;
            # we run the actual Q-network backward here, in full fp32.
            use_agent = (
                getattr(self.args, 'use_medsa', False)
                or getattr(self.args, 'use_cpc_uga', False)
            )
            if use_agent and hasattr(self, 'type_policy'):
                lam_rl = getattr(self, '_last_lam_rl', 0.)
                if lam_rl > 0:
                    dqn_loss_val = self.type_policy.update(
                        self.replay_buffer,
                        batch_size=getattr(self.args, 'dqn_batch_size', 64),
                        gamma=getattr(self.args, 'dqn_gamma', 0.99),
                        device=self.args.device,
                        optimizer=self.policy_optimizer,
                        target_update_freq=getattr(self.args, 'dqn_target_update_freq', 100),
                    )
                    lc = getattr(self, '_train_loss_components', {})
                    lc['dqn'] = dqn_loss_val * lam_rl

            # ── Per-component loss logging ────────────────────────────────
            total_val = loss_summary[-1].flatten()[0]
            lc = getattr(self, '_train_loss_components', {})

            # Original framework losses
            orig = (f"seg={self._fmt(lc.get('seg', 0))}"
                    f" bnd={self._fmt(lc.get('boundary', 0))}"
                    f" dice_reg={self._fmt(lc.get('dice_reg', 0))}"
                    + (f" refine={self._fmt(lc.get('refine', 0))}" if lc.get('refine', 0) > 0 else ""))

            # ── Agent diagnostics (MedSA / CPC–UGA) ───────────────────────
            agent_log = ""
            if getattr(self.args, 'use_cpc_uga', False):
                lam_rl  = lc.get('_lam_rl', 0.)
                eps     = lc.get('_epsilon', 0.)
                dqn_l   = lc.get('dqn',      0.)
                reward  = lc.get('reward',    0.)
                u_mean  = lc.get('_uncert',  0.)
                act_str = (f"pt={lc.get('_act_pt',0)}"
                           f" bx={lc.get('_act_bx',0)}"
                           f" sc={lc.get('_act_sc',0)}"
                           f" st={lc.get('_act_st',0)}")
                if lam_rl > 0:
                    rl_str = (f" dqn_loss={self._fmt(dqn_l)}"
                              f" r/ep={reward:+.3f}"
                              f" ε={eps:.2f}"
                              f" u={u_mean:.2f}"
                              f" [{act_str}]")
                else:
                    rl_str = (f" rl=off(ep>{getattr(self.args,'cpc_rl_start_epoch',30)})"
                              f" u={u_mean:.2f}")
                agent_log = f" | CPC-UGA:{rl_str}"
            elif getattr(self.args, 'use_medsa', False):
                lam_s   = lc.get('_lam_s',  0.)
                lam_rl  = lc.get('_lam_rl', 0.)
                eps     = lc.get('_epsilon', 0.)
                sp_loss = lc.get('spatial',  0.)
                dqn_l   = lc.get('dqn',      0.)
                reward  = lc.get('reward',    0.)   # already divided by actual_iters

                act_str = (f"pt={lc.get('_act_pt',0)}"
                           f" bx={lc.get('_act_bx',0)}"
                           f" sc={lc.get('_act_sc',0)}"
                           f" st={lc.get('_act_st',0)}")

                if lam_rl > 0:
                    rl_str = (f" dqn_loss={self._fmt(dqn_l)}"
                              f" r/ep={reward:+.3f}"
                              f" ε={eps:.2f}"
                              f" [{act_str}]")
                else:
                    rl_str = f" rl=off(ep>{getattr(self.args,'medsa_rl_start_epoch',60)})"

                sp_str = (f" spatial={self._fmt(sp_loss)}(w={lam_s:.2f})"
                          if sp_loss > 0
                          else f" spatial=off(w={lam_s:.3f})")

                agent_log = f" | MEDSA:{sp_str}{rl_str}"

            log_line = (f"epoch: {epoch_num}/{self.args.max_epoch},"
                        f" iter: {idx}/{len(self.train_data)}"
                        f" | total={self._fmt(total_val)}"
                        f" | {orig}{agent_log}")
            print(log_line)
            self.logger.info(log_line)

        print('current lr: {}'.format(self.optimizer.param_groups[0]["lr"]))
        self.update_lr(epoch_num, warm_up=self.args.warm_up)
        self.logger.info("- Train metrics: " + str(np.mean(loss_summary)))

    def validate_sliding_window(self, epoch_num):
        self.sam.eval()
        with torch.no_grad():
            dice_list = []
            for idx, (subject_dict, image_path, *_) in enumerate(self.val_data):
                label_vol = _tio_volume_4d(subject_dict['label']['data'][0])
                if label_vol.sum() <= 0:
                    self.logger.info(
                        f'{image_path} label volume too small, and it has been skipped for validation')
                    continue
                mean_dice = 0
                subject = _pad_subject_for_grid(_subject_from_dict(subject_dict), patch_size=128)
                grid_sampler = tio.inference.GridSampler(subject, 128, 16)
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

                masks_final = torch.zeros([self.args.iter_nums, len(patch_loader), 128, 128, 128])
                location_list = []
                for idx_patch, patches_batch in enumerate(patch_loader):
                    image, label = patches_batch['image'][tio.DATA].to(self.args.device), patches_batch['label'][tio.DATA].to(self.args.device)
                    locations = patches_batch[tio.LOCATION]

                    if torch.count_nonzero(label) == 0:
                        print('found empty patch')
                        masks = torch.zeros([self.args.iter_nums, 1, 128, 128, 128])
                    else:
                        _, masks = self.forward(self.sam, image, label, iter_nums=self.args.iter_nums, train=False, return_each_iter=True)
                        print(masks.shape)
                    masks_final[:, idx_patch, :] = masks.squeeze(1)
                    location_list.append(locations)

                mean_dice_sub_list = []
                for iter_num in range(self.args.iter_nums):
                    # Fresh aggregator per interaction step (do not accumulate across iters).
                    aggregator = tio.inference.GridAggregator(grid_sampler)
                    for l_i in range(0, len(location_list)):
                        location = location_list[l_i]
                        mask = masks_final[iter_num, l_i, :].unsqueeze(0).unsqueeze(0)
                        aggregator.add_batch(mask, location)
                    masks_iter_final = aggregator.get_output_tensor()
                    mean_dice_sub_list.append(self.get_dice_score(torch.sigmoid(masks_iter_final), subject.label.data))

                mean_dice_sub = np.mean(mean_dice_sub_list)
                mean_dice += mean_dice_sub
                dice_list.append(mean_dice)
                print(mean_dice_sub)
                self.logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.val_data)) +
                    ' subject: ' + str(image_path) + ' mean dice over clicks:' + str(mean_dice) +
                    ' stich left and right side (total size): ' + str(subject.label.data.size(0)))
            self.logger.info("- Val metrics mean dice: " + str(np.mean(dice_list)))
        return dice_list

    def validate(self, epoch_num):
        self.sam.eval()
        device = self.args.device
        with torch.no_grad():
            dice_list       = []
            iters_used_list = []   # actual iterations used by CORA policy

            for idx, (image, label, image_path, _) in enumerate(self.val_data):
                mean_dice = 0
                image, label = image.to(device), label.to(device)

                if self.args.data == 'kits' and image.size(1) > 1:
                    label_final  = torch.zeros([1, 1, int(image.size(2) * 2), image.size(3), image.size(4)])
                    masks_final  = torch.zeros([self.args.iter_nums, 1, int(image.size(2) * 2), image.size(3), image.size(4)])

                    for channel_num in range(image.size(1)):
                        _, masks = self.forward(self.sam, image[:, channel_num, :].unsqueeze(1),
                                                label[:, channel_num, :].unsqueeze(1),
                                                iter_nums=self.args.iter_nums, train=False, return_each_iter=True)
                        start_point = 0 + channel_num * image.size(2)
                        end_pont    = image.size(2) + channel_num * image.size(2)
                        masks_final[:, 0, start_point: end_pont, :] = masks[:, 0, :]
                        label_final[0, 0, start_point: end_pont, :]  = label[0, channel_num, :]

                    mean_dice_sub_list = []
                    for iter_num in range(self.args.iter_nums):
                        mean_dice_sub_list.append(self.get_dice_score(torch.sigmoid(masks_final[iter_num]), label_final[0]))
                    mean_dice_sub = np.mean(mean_dice_sub_list)
                else:
                    mean_dice_sub, masks = self.forward(self.sam, image, label,
                                                        iter_nums=self.args.iter_nums, train=False)

                mean_dice += mean_dice_sub
                dice_list.append(mean_dice)

                if hasattr(self, '_clinical_iters_used'):
                    iters_used_list.append(self._clinical_iters_used)

                print(mean_dice_sub)
                self.logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.val_data)) +
                    ' subject: ' + str(image_path) + ' mean dice over clicks:' + str(mean_dice) +
                    ' stich left and right side (total size): ' + str(label.size(1)))

            self.logger.info("- Val metrics mean dice: " + str(np.mean(dice_list)))
            if iters_used_list:
                self.logger.info(
                    f"- Val CORA | avg_iters_used: {np.mean(iters_used_list):.2f}"
                )

        return dice_list


    # ── Clinical utility helpers ───────────────────────────────────────────────

    @torch.no_grad()
    def _compute_batch_dice(self, pred_prob, label):
        """Per-sample Dice score. Returns tensor (B,).
        Works regardless of whether masks have a channel dimension."""
        B = pred_prob.size(0)
        pred_bin = (pred_prob > 0.5).float().reshape(B, -1)   # (B, N)
        gt_bin   = (label > 0).float().reshape(B, -1)         # (B, N)
        intersection = (pred_bin * gt_bin).sum(dim=1)        # (B,)
        union        = pred_bin.sum(dim=1) + gt_bin.sum(dim=1)  # (B,)
        return 2.0 * intersection / (union + 1e-8)           # (B,)

    # ─────────────────────────────────────────────────────────────────────────

    def calculate_loss(self, mask, prev_masks, pred_dice, label, labels_input, iter_num,
                       inter=False):
        """
        Computes total loss for one candidate mask.

        Returns:
            (total_loss, loss_dict)
            loss_dict keys: 'seg', 'boundary', 'dice_reg'
            All values are plain Python floats for logging.
        """
        mask_probs = torch.sigmoid(mask)

        seg_edge  = abs(label - self.pooling_layer(label))
        mask_edge = abs(mask_probs - self.pooling_layer(mask_probs))

        l_seg      = self.loss_segmentation(mask, label)
        l_boundary = self.loss_boundary(mask_edge, seg_edge) * 10

        l_dice_reg = torch.tensor(0.0, device=mask.device)
        for batch_index in range(mask.size(0)):
            target_dice = 1 - self.loss_validation(
                mask[batch_index].unsqueeze(0), label[batch_index].unsqueeze(0)
            )[0, 0, 0, 0, 0]
            target_dice = torch.tensor([target_dice])[0].to(self.args.device)
            l_dice_reg  = l_dice_reg + self.loss_boundary(pred_dice[batch_index], target_dice)

        loss = l_seg + l_boundary + l_dice_reg

        loss_dict = {
            'seg':      l_seg.item(),
            'boundary': l_boundary.item(),
            'dice_reg': l_dice_reg.item(),
        }
        return loss, loss_dict

    def get_dice_score(self, prev_masks, label, batch=False):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.nan
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (label > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        if batch:
            return dice_list
        else:
            valid = [d.item() if hasattr(d, 'item') else float(d) for d in dice_list]
            valid = [d for d in valid if not np.isnan(d)]
            return float(np.mean(valid)) if valid else 0.0

    def save_model(self, current_dice, epoch_num):
        is_best = False
        if np.mean(current_dice) > self.best_dice:
            self.best_dice = np.mean(current_dice)
            self.best_epoch = epoch_num
            is_best = True

        if not self.args.ddp or (self.args.ddp and self.args.rank == 0):
            ckpt_dict = {
                "epoch": epoch_num + 1,
                "best_val_loss": self.best_dice,
                "model_state_dict": self.sam.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            }
            if getattr(self.args, 'use_medsa', False) and hasattr(self, 'edit_encoder'):
                ckpt_dict['medsa_state'] = {
                    'edit_encoder':     self.edit_encoder.state_dict(),
                    'memory':           self.memory.state_dict(),
                    'type_policy':      self.type_policy.state_dict(),
                    'policy_optimizer': self.policy_optimizer.state_dict(),
                }
            if getattr(self.args, 'use_cpc_uga', False) and hasattr(self, 'type_policy'):
                ckpt_dict['cpc_uga_state'] = {
                    'type_policy':      self.type_policy.state_dict(),
                    'policy_optimizer': self.policy_optimizer.state_dict(),
                }
            save_checkpoint(ckpt_dict, is_best=is_best, checkpoint=self.args.save_dir)
        self.logger.info("- Val metrics best mean dice: {} at epoch {} " .format(self.best_dice, self.best_epoch))

    def setup(self):
        self.setup_loss()
        self.setup_optimizier()
        self.setup_scheduler()

        if self.args.resume:
            if self.args.ddp:
                dist.barrier()
            checkpoint = 'best.pth.tar' if self.args.resume_best else 'last.pth.tar'
            ckpt = torch.load(os.path.join(self.args.save_dir, checkpoint), weights_only=False)

            self.start_epoch = ckpt["epoch"]
            self.best_epoch = self.start_epoch
            self.best_dice = ckpt["best_val_loss"]
            # strict=False: gracefully ignore keys removed from the architecture
            # (e.g. acceptance_head / effort_head) when resuming from older checkpoints.
            missing, unexpected = self.sam.load_state_dict(ckpt["model_state_dict"], strict=False)
            if unexpected:
                self.logger.warning(f"Checkpoint keys ignored (removed from model): {unexpected}")
            if missing:
                self.logger.warning(f"Model keys missing in checkpoint (new layers, random init): {missing}")
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                self.logger.warning(
                    "Optimizer state dict incompatible with current model "
                    "(architecture changed). Starting with a fresh optimizer."
                )
            try:
                self.lr_scheduler_regular.load_state_dict(ckpt['lr_scheduler'])
            except Exception:
                self.logger.warning("LR scheduler state incompatible — resetting scheduler.")

            if getattr(self.args, 'use_medsa', False) and 'medsa_state' in ckpt:
                ms = ckpt['medsa_state']
                if hasattr(self, 'edit_encoder'):
                    self.edit_encoder.load_state_dict(ms['edit_encoder'])
                if hasattr(self, 'memory'):
                    self.memory.load_state_dict(ms['memory'])
                if hasattr(self, 'type_policy'):
                    self.type_policy.load_state_dict(ms['type_policy'])
                try:
                    if hasattr(self, 'policy_optimizer'):
                        self.policy_optimizer.load_state_dict(ms['policy_optimizer'])
                except Exception:
                    self.logger.warning("Policy optimizer state incompatible — resetting.")
                self.logger.info("Resumed MedSA component weights from checkpoint.")
            elif getattr(self.args, 'use_medsa', False):
                self.logger.warning("MedSA enabled but no medsa_state in checkpoint — using random init.")

            if getattr(self.args, 'use_cpc_uga', False) and 'cpc_uga_state' in ckpt:
                cs = ckpt['cpc_uga_state']
                if hasattr(self, 'type_policy'):
                    self.type_policy.load_state_dict(cs['type_policy'])
                try:
                    if hasattr(self, 'policy_optimizer'):
                        self.policy_optimizer.load_state_dict(cs['policy_optimizer'])
                except Exception:
                    self.logger.warning("CPC–UGA policy optimizer incompatible — resetting.")
                self.logger.info("Resumed CPC–UGA policy weights from checkpoint.")
            elif getattr(self.args, 'use_cpc_uga', False):
                self.logger.warning("CPC–UGA enabled but no cpc_uga_state in checkpoint — random init.")

            self.logger.info(f"Resume training from epoch {self.start_epoch}!")
            del ckpt
            torch.cuda.empty_cache()

    def setup_loss(self):
        self.loss_boundary = nn.MSELoss()
        self.mse_none = nn.MSELoss(reduction='none')

        self.loss_segmentation = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.loss_Dice = DiceLoss(sigmoid=True)
        self.loss_validation = DiceLoss(sigmoid=True, reduction='none')

        self.l1 = nn.L1Loss()
        self.inter_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        # Clinical utility heads losses

        # MEDSA Spatial Next-Prompt Head loss
        self.loss_spatial = nn.MSELoss()

    def setup_optimizier(self):
        self.optimizer = AdamW([
            {'params': self.sam.image_encoder.parameters()},
            {'params': self.sam.prompt_encoder.parameters()},
            {'params': self.sam.mask_decoder.parameters()},
        ], lr=self.args.lr)

    def setup_scheduler(self):
        if self.args.lr_scheduler == 'linear':
            self.lr_scheduler_regular = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=500)
        else:
            self.lr_scheduler_regular = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        if self.args.warm_up:
            self.linear_warmup_scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)

    def update_lr(self, epoch, warmup_epoch=10, warm_up=False):
        if warm_up:
            if epoch < warmup_epoch:
                self.lr_scheduler = self.linear_warmup_scheduler
            else:
                self.lr_scheduler = self.lr_scheduler_regular
        else:
            self.lr_scheduler = self.lr_scheduler_regular
        self.lr_scheduler.step()










