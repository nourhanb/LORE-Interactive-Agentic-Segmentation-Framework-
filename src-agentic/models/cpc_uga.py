"""
Clinical Priority Correction + Uncertainty-Gated Accept (CPC–UGA)
================================================================
SPIE interactive loop — deliberately NOT CoCC / MedSA:

  1. Clinical Priority Ranker  — which error region to correct
  2. Mask Uncertainty         — disagreement among candidate masks (+ IoU entropy)
  3. Tiny DQN / gated decision — accept vs point / box / scribble

No edit-memory encoder, no next-click heatmap, no spatial-grounding ξ.
"""

from __future__ import annotations

import math
import random
import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.policy import (
    ACTION_POINT, ACTION_BOX, ACTION_SCRIBBLE, ACTION_STOP,
    N_ACTIONS, ACTION_EFFORT, ReplayBuffer, STOP_QUALITY_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mask uncertainty (current segmentation — not next-click localization)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def mask_uncertainty(
    masks: torch.Tensor,                 # (B, N, X, Y, Z) logits
    iou_pred: Optional[torch.Tensor] = None,  # (B, N) optional quality scores
) -> float:
    """
    Scalar uncertainty ∈ [0, 1] from:
      • voxel-wise std across candidate masks (disagreement)
      • entropy of softmax(IoU head) over candidates
    """
    if masks is None or masks.numel() == 0:
        return 1.0

    probs = torch.sigmoid(masks.float())          # (B, N, ...)
    N = probs.size(1)
    if N <= 1:
        u_dis = 0.5
    else:
        # Mean absolute deviation from the mean candidate (bounded ~[0, 0.5])
        mean_p = probs.mean(dim=1, keepdim=True)
        mad = (probs - mean_p).abs().mean().item()
        u_dis = min(1.0, mad / 0.25)              # saturate near 0.25 MAD

    if iou_pred is not None and iou_pred.numel() > 1:
        p = F.softmax(iou_pred.float(), dim=1).clamp_min(1e-8)
        ent = -(p * p.log()).sum(dim=1).mean().item()
        u_iou = ent / math.log(float(N))
    else:
        u_iou = 0.5

    return float(np.clip(0.5 * u_dis + 0.5 * u_iou, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Tiny policy (5-D state — no CoCC embeddings)
# ─────────────────────────────────────────────────────────────────────────────

class _MLP(nn.Sequential):
    def __init__(self, dims: List[int]):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class PriorityUncertaintyPolicy(nn.Module):
    """
    State (dim 5):
      1. dice_current
      2. delta_dice
      3. iter_progress
      4. priority_volume   — fraction of volume in the top clinical-priority CC
      5. mask_uncertainty  — candidate-mask disagreement (+ IoU entropy)
    """
    STATE_DIM = 5

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dims: tuple = (32, 16),
        n_actions: int = N_ACTIONS,
        soft_update_tau: float = 0.005,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.soft_update_tau = soft_update_tau
        dims = [state_dim] + list(hidden_dims) + [n_actions]
        self.q_net = _MLP(dims)
        self.target_net = copy.deepcopy(self.q_net)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self._update_step = 0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_net(state)

    @staticmethod
    def build_state(
        dice_current: float,
        delta_dice: float,
        iter_progress: float,
        priority_volume: float,
        uncertainty: float,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.tensor(
            [dice_current, delta_dice, iter_progress, priority_volume, uncertainty],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)

    def select_action(
        self,
        state: torch.Tensor,
        epsilon: float,
        priority_volume: float,
        boundary_overlap: float,
        iter_idx: int,
        max_iters: int,
        dice_current: float = 0.0,
        uncertainty: float = 1.0,
        uncert_low: float = 0.35,
        uncert_high: float = 0.65,
    ) -> int:
        """ε-greedy with uncertainty-gated accept / escalate hard masks."""
        auto_accept = uncertainty < uncert_low and dice_current >= STOP_QUALITY_THRESHOLD
        escalate = uncertainty > uncert_high

        def _escalate_action() -> int:
            if priority_volume > 0.05:
                return ACTION_BOX
            if boundary_overlap > 0.30 and priority_volume <= 0.15:
                return ACTION_SCRIBBLE
            return ACTION_POINT

        if random.random() < epsilon:
            action = random.randint(0, self.n_actions - 1)
            if action == ACTION_STOP and dice_current < STOP_QUALITY_THRESHOLD:
                action = ACTION_POINT
            if escalate and action == ACTION_STOP:
                action = _escalate_action()
            if auto_accept and random.random() < 0.5:
                action = ACTION_STOP
            return action

        with torch.no_grad():
            q = self.q_net(state).float()
            if priority_volume > 0.15:
                q[:, ACTION_SCRIBBLE] = -1e4
            if dice_current < STOP_QUALITY_THRESHOLD:
                q[:, ACTION_STOP] = -1e4
            elif iter_idx >= max_iters - 3:
                q[:, ACTION_STOP] += 2.0

            if escalate:
                q[:, ACTION_STOP] = -1e4
                # Soft bias toward geometry-appropriate escalate action
                pref = _escalate_action()
                q[:, pref] += 2.0
                if pref != ACTION_BOX:
                    q[:, ACTION_BOX] += 0.5
            elif auto_accept:
                q[:, ACTION_STOP] += 2.5

            return int(q.argmax(dim=-1).item())

    def update(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        gamma: float,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        target_update_freq: int = 100,
    ) -> float:
        if len(replay_buffer) < batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states_t = torch.from_numpy(states).to(device)
        actions_t = torch.from_numpy(actions).long().to(device)
        rewards_t = torch.from_numpy(rewards).to(device)
        next_states_t = torch.from_numpy(next_states).to(device)
        dones_t = torch.from_numpy(dones).to(device)

        if (torch.isnan(states_t).any() or torch.isnan(rewards_t).any() or
                torch.isnan(next_states_t).any()):
            replay_buffer.buffer.clear()
            return 0.0

        q_current = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            q_target = rewards_t + gamma * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_current, q_target)
        if torch.isnan(loss):
            return 0.0
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        optimizer.step()

        self._update_step += 1
        if self._update_step % target_update_freq == 0:
            tau = self.soft_update_tau
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
        return float(loss.item())


def compute_cpc_reward(
    action: int,
    dice_current: float,
    dice_prev: float,
    priority_volume: float,
    boundary_overlap: float,
    priority_hit: float = 0.0,
    uncertainty: float = 0.0,
    beta_priority: float = 0.30,
    gamma_uncert: float = 0.20,
) -> float:
    """
    R = ΔDice − effort·(1+prio_vol) + scribble boundary bonus
        + r_stop + β·priority_hit·[a≠stop] + uncertainty gate terms.
    No spatial-grounding ξ.
    """
    delta_dice = dice_current - dice_prev
    effort = float(ACTION_EFFORT[action].item())
    r = delta_dice
    r -= effort * (1.0 + priority_volume)
    if action == ACTION_SCRIBBLE:
        r += 0.3 * boundary_overlap
    if action == ACTION_STOP:
        r += (dice_current - STOP_QUALITY_THRESHOLD)
        if dice_current < 0.60:
            r -= 0.5
        r -= gamma_uncert * uncertainty
    else:
        r += beta_priority * priority_hit
        if action in (ACTION_BOX, ACTION_SCRIBBLE):
            r += gamma_uncert * uncertainty
    return float(np.clip(r, -3.0, 3.0))
