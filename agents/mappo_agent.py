"""
agents/mappo_agent.py

MAPPO Actor-Critic network.

Architecture:
  Actor  (decentralized): obs(65) → MLP → Discrete(2)  [BID / PASS]
  Critic (centralized):   global_state(650) → MLP → V(s)

Centralized Training, Decentralized Execution (CTDE):
  - During training, critic sees ALL agents' observations concatenated
  - During execution, actor only sees its own 65-dim observation
  - All agents share one set of weights (parameter sharing)
    → faster convergence, better generalization across teams

Key MAPPO tricks implemented:
  1. Value normalization  — running mean/std on returns
  2. Orthogonal init      — better gradient flow at start
  3. GAE advantages       — smoother credit assignment
  4. Clipped PPO objective
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional

OBS_DIM          = 65
GLOBAL_STATE_DIM = 650   # 10 agents × 65
ACTION_DIM       = 2     # BID or PASS


# ── Weight initialisation ─────────────────────────────────────────

def orthogonal_init(module: nn.Module, gain: float = 1.0):
    """Orthogonal initialisation — key for stable PPO training."""
    if isinstance(module, (nn.Linear,)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_dim: int,
    activation: str = "tanh",
    output_gain: float = 1.0,
    use_orthogonal: bool = True,
) -> nn.Sequential:
    act_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()
    layers = []
    in_dim = input_dim
    for _ in range(num_layers):
        linear = nn.Linear(in_dim, hidden_dim)
        if use_orthogonal:
            orthogonal_init(linear, gain=np.sqrt(2))
        layers += [linear, act_fn]
        in_dim = hidden_dim
    out = nn.Linear(hidden_dim, output_dim)
    if use_orthogonal:
        orthogonal_init(out, gain=output_gain)
    layers.append(out)
    return nn.Sequential(*layers)


# ── Value normalizer ──────────────────────────────────────────────

class RunningMeanStd:
    """
    Tracks running mean and variance of value targets.
    Used to normalise returns for stable value learning.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.mean    = 0.0
        self.var     = 1.0
        self.count   = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x)
        batch_var  = np.var(x)
        batch_count = x.size
        delta   = batch_mean - self.mean
        tot     = self.count + batch_count
        self.mean  = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2  = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var   = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * np.sqrt(self.var) + self.mean


# ── Actor network ─────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Decentralised actor — takes own obs (65,) → action logits (2,).
    Shared across all 10 agents (parameter sharing).
    """

    def __init__(
        self,
        obs_dim:        int   = OBS_DIM,
        hidden_dim:     int   = 128,
        num_layers:     int   = 2,
        activation:     str   = "tanh",
        use_orthogonal: bool  = True,
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=ACTION_DIM,
            activation=activation,
            output_gain=0.01,   # small init → near-uniform policy at start
            use_orthogonal=use_orthogonal,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns action logits."""
        return self.net(obs)

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.forward(obs)
        return Categorical(logits=logits)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample action and return (action, log_prob, entropy).
        """
        dist     = self.get_distribution(obs)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Used during PPO update — get log_prob and entropy for stored actions."""
        dist     = self.get_distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, entropy


# ── Critic network ────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Centralised critic — takes global state (650,) → scalar value V(s).
    Only used during training, not execution.
    Global state = concatenation of all 10 agents' obs vectors.
    """

    def __init__(
        self,
        global_state_dim: int  = GLOBAL_STATE_DIM,
        hidden_dim:       int  = 256,
        num_layers:       int  = 2,
        activation:       str  = "tanh",
        use_orthogonal:   bool = True,
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=global_state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            activation=activation,
            output_gain=1.0,
            use_orthogonal=use_orthogonal,
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Returns value estimate V(s), shape (batch, 1)."""
        return self.net(global_state)


# ── Combined MAPPO agent ──────────────────────────────────────────

class MAPPOAgent(nn.Module):
    """
    Full MAPPO agent wrapping Actor + Critic.
    One instance shared across all 10 teams (parameter sharing).
    """

    def __init__(
        self,
        obs_dim:          int   = OBS_DIM,
        global_state_dim: int   = GLOBAL_STATE_DIM,
        hidden_dim:       int   = 128,
        num_layers:       int   = 2,
        activation:       str   = "tanh",
        use_orthogonal:   bool  = True,
        value_normalize:  bool  = True,
        device:           str   = "cpu",
    ):
        super().__init__()
        self.device          = torch.device(device)
        self.value_normalize = value_normalize

        self.actor  = Actor(obs_dim, hidden_dim, num_layers,
                            activation, use_orthogonal)
        self.critic = Critic(global_state_dim, hidden_dim * 2, num_layers,
                             activation, use_orthogonal)

        self.value_normalizer = RunningMeanStd() if value_normalize else None
        self.to(self.device)

    # ── Execution (decentralised) ─────────────────────────────────

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        """
        Called during rollout collection for each agent's turn.
        obs: (65,) numpy array
        Returns: action (int), log_prob (float)
        """
        obs_t    = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, _ = self.actor.act(obs_t, deterministic)
        return action.item(), log_prob.item()

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        """
        Called during rollout for value bootstrap.
        global_state: (650,) numpy array
        """
        gs_t  = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        value = self.critic(gs_t)
        if self.value_normalize and self.value_normalizer:
            value = torch.FloatTensor(
                self.value_normalizer.denormalize(value.cpu().numpy())
            )
        return value.item()

    # ── Training (centralised) ────────────────────────────────────

    def evaluate_for_update(
        self,
        obs_batch:          torch.Tensor,
        global_state_batch: torch.Tensor,
        actions_batch:      torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used inside PPO update.
        Returns: log_probs, values, entropy
        """
        log_probs, entropy = self.actor.evaluate_actions(obs_batch, actions_batch)
        values             = self.critic(global_state_batch).squeeze(-1)

        if self.value_normalize and self.value_normalizer:
            # Denormalize before returning (advantages computed in raw space)
            values_raw = torch.FloatTensor(
                self.value_normalizer.denormalize(values.detach().cpu().numpy())
            ).to(self.device)
        else:
            values_raw = values

        return log_probs, values_raw, entropy

    def normalize_values(self, values: np.ndarray) -> np.ndarray:
        if self.value_normalize and self.value_normalizer:
            self.value_normalizer.update(values)
            return self.value_normalizer.normalize(values)
        return values

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "value_normalizer_mean":  getattr(self.value_normalizer, "mean", 0),
            "value_normalizer_var":   getattr(self.value_normalizer, "var", 1),
            "value_normalizer_count": getattr(self.value_normalizer, "count", 1e-8),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if self.value_normalizer:
            self.value_normalizer.mean  = ckpt.get("value_normalizer_mean", 0)
            self.value_normalizer.var   = ckpt.get("value_normalizer_var", 1)
            self.value_normalizer.count = ckpt.get("value_normalizer_count", 1e-8)
        print(f"  Loaded checkpoint: {path}")
