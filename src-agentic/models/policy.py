"""
MEDSA Edit-Conditioned Type Policy
====================================
Components:
  EditConditionedTypePolicy  — DQN Q-network over {point, box, scribble, stop}.
  ReplayBuffer               — Circular experience replay buffer.

Action space:
  0 = point    (effort 0.1)
  1 = box      (effort 0.3)
  2 = scribble (effort 0.5)
  3 = stop     (effort 0.0)
"""

import random
import copy
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Action constants
# ─────────────────────────────────────────────────────────────────────────────

ACTION_POINT    = 0
ACTION_BOX      = 1
ACTION_SCRIBBLE = 2
ACTION_STOP     = 3
N_ACTIONS       = 4

# Cognitive effort proxy per action (matches reward formula)
ACTION_EFFORT = torch.tensor([0.1, 0.3, 0.5, 0.0], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular replay buffer for DQN experience replay."""

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# DQN Q-network
# ─────────────────────────────────────────────────────────────────────────────

class _MLP(nn.Sequential):
    def __init__(self, dims: List[int]):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class EditConditionedTypePolicy(nn.Module):
    """
    DQN-style type policy head.

    State vector (dim 8) — compact, interpretable scalars derived from
    the encoder embeddings rather than the raw high-dimensional vectors.
    This avoids non-stationarity: the DQN state is stable even as the
    underlying encoders continue to be updated by the RL objective.

      1. dice_current    — current Dice score
      2. delta_dice      — Dice improvement at this step  (d_i − d_{i−1})
      3. iter_progress   — step fraction  i / T  ∈ [0, 1]
      4. edit_volume     — v_i  (normalised correction volume)
      5. edit_bnd_ovlp   — b_i  (boundary overlap of error)
      6. error_magnitude — ‖e_cnn‖ / (‖e_cnn‖ + 1)  (error severity)
      7. persistence     — cos(e_cnn, m_cnn)  (recurring error signal)
      8. spatial_entropy — H(σ(H_i)), normalised  (localization confidence)
      ──────────────────────────────────────────────────────────────────
      total  8

    Action space: {0=point, 1=box, 2=scribble, 3=stop}
    """
    STATE_DIM = 8

    def __init__(
        self,
        state_dim:   int = STATE_DIM,
        hidden_dims: tuple = (32, 16),
        n_actions:   int = N_ACTIONS,
        soft_update_tau: float = 0.005,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.soft_update_tau = soft_update_tau

        dims = [state_dim] + list(hidden_dims) + [n_actions]
        self.q_net      = _MLP(dims)
        self.target_net = copy.deepcopy(self.q_net)
        for p in self.target_net.parameters():
            p.requires_grad = False

        self._update_step = 0

    # ── Q-value prediction ────────────────────────────────────────────────────

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, STATE_DIM) → Q-values (B, N_ACTIONS)"""
        return self.q_net(state)

    def q_target(self, state: torch.Tensor) -> torch.Tensor:
        return self.target_net(state)

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(
        self,
        state:          torch.Tensor,   # (1, STATE_DIM=8) — already detached
        epsilon:        float,
        edit_volume:    float,
        iter_idx:       int,
        max_iters:      int,
        dice_current:   float = 0.0,    # hard-mask: forbid stop below quality threshold
        spatial_entropy: float = 1.0,
        persistence:    float = 0.0,
        use_uncertainty_gate: bool = False,
        uncert_low:     float = 0.35,
        uncert_high:    float = 0.65,
        persist_thresh: float = 0.85,
    ) -> int:
        """ε-greedy with physician-safety hard masks (+ optional uncertainty gate)."""
        escalate = (
            use_uncertainty_gate
            and (spatial_entropy > uncert_high or persistence >= persist_thresh)
        )
        auto_accept = (
            use_uncertainty_gate
            and spatial_entropy < uncert_low
            and dice_current >= STOP_QUALITY_THRESHOLD
        )

        if random.random() < epsilon:
            action = random.randint(0, self.n_actions - 1)
            # Even during exploration: never stop below the quality floor
            if action == ACTION_STOP and dice_current < STOP_QUALITY_THRESHOLD:
                action = ACTION_POINT
            # Uncertainty gate during exploration
            if escalate and action == ACTION_STOP:
                action = ACTION_BOX if edit_volume > 0.15 else ACTION_SCRIBBLE
            if auto_accept and action != ACTION_STOP and dice_current >= STOP_QUALITY_THRESHOLD:
                # Occasionally honour auto-accept even under ε (50%)
                if random.random() < 0.5:
                    action = ACTION_STOP
        else:
            with torch.no_grad():
                q = self.q_net(state).float()           # (1, N_ACTIONS) — cast to fp32 to avoid fp16 overflow in masks
                # Hard masks ─────────────────────────────
                if edit_volume > 0.15:
                    q[:, ACTION_SCRIBBLE] = -1e4        # fatigue: suppress scribble
                if dice_current < STOP_QUALITY_THRESHOLD:
                    q[:, ACTION_STOP] = -1e4            # quality floor: forbid early stop
                elif iter_idx >= max_iters - 3:
                    q[:, ACTION_STOP] += 2.0            # near end: nudge towards stop

                # Uncertainty-gated escalate vs auto-accept
                if escalate:
                    q[:, ACTION_STOP] = -1e4
                    q[:, ACTION_BOX] += 1.5
                    if edit_volume <= 0.15:
                        q[:, ACTION_SCRIBBLE] += 1.0
                elif auto_accept:
                    q[:, ACTION_STOP] += 2.5

                action = int(q.argmax(dim=-1).item())
        return action

    # ── State tensor builder ──────────────────────────────────────────────────

    @staticmethod
    def build_state(
        dice_current:    float,
        delta_dice:      float,
        iter_progress:   float,          # iter_idx / max_iters ∈ [0, 1]
        edit_volume:     float,          # v_i
        edit_bnd_ovlp:   float,          # b_i
        error_magnitude: float,          # ‖e_cnn‖ / (‖e_cnn‖ + 1)
        persistence:     float,          # cos(e_cnn, m_cnn)
        spatial_entropy: float,          # H(σ(H_i)), normalised ∈ [0, 1]
        device:          torch.device,
    ) -> torch.Tensor:                   # (1, 8)
        return torch.tensor(
            [dice_current, delta_dice, iter_progress,
             edit_volume, edit_bnd_ovlp,
             error_magnitude, persistence, spatial_entropy],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)

    # ── DQN training step ─────────────────────────────────────────────────────

    def update(
        self,
        replay_buffer: "ReplayBuffer",
        batch_size:    int,
        gamma:         float,
        device:        torch.device,
        optimizer:     torch.optim.Optimizer,
        target_update_freq: int = 100,
    ) -> float:
        """Sample a mini-batch from replay buffer and perform one DQN step."""
        if len(replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states_t      = torch.from_numpy(states).to(device)
        actions_t     = torch.from_numpy(actions).long().to(device)
        rewards_t     = torch.from_numpy(rewards).to(device)
        next_states_t = torch.from_numpy(next_states).to(device)
        dones_t       = torch.from_numpy(dones).to(device)

        # NaN guard: if any NaN has leaked into the sampled batch, flush the
        # entire replay buffer and skip this update to prevent Q-value divergence.
        if (torch.isnan(states_t).any() or torch.isnan(rewards_t).any() or
                torch.isnan(next_states_t).any()):
            replay_buffer.buffer.clear()
            return 0.0

        # Current Q
        q_vals    = self.q_net(states_t)
        q_current = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q (Double DQN)
        with torch.no_grad():
            next_actions   = self.q_net(next_states_t).argmax(dim=1)
            next_q         = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target_val   = rewards_t + gamma * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_current, q_target_val)
        if torch.isnan(loss):
            return 0.0
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        optimizer.step()

        # Soft update target network
        self._update_step += 1
        if self._update_step % target_update_freq == 0:
            self._soft_update()

        return float(loss.item())

    def _soft_update(self):
        tau = self.soft_update_tau
        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)


# ─────────────────────────────────────────────────────────────────────────────
# Reward computation (stateless helper)
# ─────────────────────────────────────────────────────────────────────────────

STOP_QUALITY_THRESHOLD = 0.70  # minimum acceptable DSC for rewarding a stop

def compute_dqn_reward(
    action:        int,
    dice_current:  float,
    dice_prev:     float,
    edit_volume:   float,
    edit_bnd_ovlp: float,
    spatial_hit:   float = 0.0,   # ∈ [0,1] — optional residual spatial grounding
    priority_hit:  float = 0.0,   # ∈ [0,1] — fraction of edit on high-priority CC
    spatial_entropy: float = 0.0, # ∈ [0,1] — uncertainty of H_{i-1}
    use_clinical_priority: bool = False,
    use_uncertainty_gate: bool = False,
    beta_priority: float = 0.30,
    gamma_uncert:  float = 0.20,
    xi_weight:     float = 0.10,  # down-weighted residual ξ when priority is on
) -> float:
    """
    Reward for clinical-priority + uncertainty-gated interactive segmentation.

    R_i = ΔDice
          − effort(a) · (1 + edit_volume)
          + 0.3 · edit_bnd_ovlp · [a == scribble]
          + r_stop(a, d)
          + β · priority_hit · [a != stop]          ← clinical priority (headline)
          + γ · uncertainty gate terms
          + ξ_weight · spatial_hit · [a != stop]    ← residual heatmap learning cue

    When clinical priority / uncertainty gate flags are off, falls back to the
    legacy LORE-style reward (ξ weight 0.25, no priority / gate terms).
    """
    delta_dice = dice_current - dice_prev
    effort     = float(ACTION_EFFORT[action].item())

    # Legacy ξ weight when new modules are disabled
    if not use_clinical_priority and not use_uncertainty_gate:
        xi_weight = 0.25
        beta_priority = 0.0
        gamma_uncert = 0.0

    r  = delta_dice
    r -= effort * (1.0 + edit_volume)
    if action == ACTION_SCRIBBLE:
        r += 0.3 * edit_bnd_ovlp
    if action == ACTION_STOP:
        r += (dice_current - STOP_QUALITY_THRESHOLD)
        if dice_current < 0.60:
            r -= 0.5
        # Penalise stopping under high spatial uncertainty
        if use_uncertainty_gate:
            r -= gamma_uncert * spatial_entropy
    else:
        if use_clinical_priority:
            r += beta_priority * priority_hit
        r += xi_weight * spatial_hit
        # Reward escalating (box/scribble) when uncertainty is high
        if use_uncertainty_gate and action in (ACTION_BOX, ACTION_SCRIBBLE):
            r += gamma_uncert * spatial_entropy
    return float(np.clip(r, -3.0, 3.0))
