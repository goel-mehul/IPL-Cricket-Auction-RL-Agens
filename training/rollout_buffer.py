"""
training/rollout_buffer.py

Stores (obs, global_state, action, reward, done, log_prob, value)
tuples collected during rollout, then computes GAE advantages
and returns for the PPO update.

One buffer per agent, but since we use parameter sharing,
all agents' data is batched together for the update.
"""

import numpy as np
from typing import Dict, List

OBS_DIM          = 65
GLOBAL_STATE_DIM = 650


class AgentBuffer:
    """Stores one agent's experience for one rollout."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.obs           = []
        self.global_states = []
        self.actions       = []
        self.rewards       = []
        self.dones         = []
        self.log_probs     = []
        self.values        = []
        self._size         = 0

    def add(self, obs, global_state, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self._size += 1

    def size(self) -> int:
        return self._size


class RolloutBuffer:
    """
    Manages rollout buffers for all 10 agents.
    After rollout, computes GAE advantages and prepares minibatches.
    """

    def __init__(self, num_agents: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.num_agents  = num_agents
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.agents      = {}   # team_id → AgentBuffer
        self._capacity   = 10000   # soft cap, reset each rollout

    def init_agent(self, team_id: str):
        self.agents[team_id] = AgentBuffer(self._capacity)

    def reset_all(self):
        for buf in self.agents.values():
            buf.reset()

    def add(self, team_id: str, obs, global_state, action, reward, done, log_prob, value):
        if team_id not in self.agents:
            self.init_agent(team_id)
        self.agents[team_id].add(obs, global_state, action, reward, done, log_prob, value)

    def total_steps(self) -> int:
        return sum(buf.size() for buf in self.agents.values())

    def compute_gae(self, team_id: str, last_value: float = 0.0) -> np.ndarray:
        """
        Generalised Advantage Estimation for one agent.

        GAE(λ) smoothly interpolates between TD(1) and Monte Carlo:
            δt = r_t + γ·V(s_{t+1}) - V(s_t)
            A_t = δt + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...

        Returns advantages array.
        """
        buf      = self.agents[team_id]
        rewards  = np.array(buf.rewards,   dtype=np.float32)
        dones    = np.array(buf.dones,     dtype=np.float32)
        values   = np.array(buf.values,    dtype=np.float32)
        n        = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0

        # Bootstrap: append last_value for terminal state
        values_ext = np.append(values, last_value)

        for t in reversed(range(n)):
            next_non_terminal = 1.0 - dones[t]
            delta    = rewards[t] + self.gamma * values_ext[t+1] * next_non_terminal - values_ext[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        return advantages

    def get_all_data(self, agent_ids: List[str]):
        """
        Collect data from all agents, compute GAE, return batched arrays.
        Since all agents share parameters, we stack everyone's data.

        Returns dict with numpy arrays ready for PPO update.
        """
        all_obs, all_gs, all_act = [], [], []
        all_adv, all_ret, all_lp = [], [], []

        for team_id in agent_ids:
            buf = self.agents.get(team_id)
            if buf is None or buf.size() == 0:
                continue

            advantages = self.compute_gae(team_id, last_value=0.0)
            values     = np.array(buf.values, dtype=np.float32)
            returns    = advantages + values   # V_target = A + V

            all_obs.append(np.array(buf.obs,           dtype=np.float32))
            all_gs.append( np.array(buf.global_states, dtype=np.float32))
            all_act.append(np.array(buf.actions,       dtype=np.int64))
            all_adv.append(advantages)
            all_ret.append(returns)
            all_lp.append( np.array(buf.log_probs,     dtype=np.float32))

        if not all_obs:
            return None

        obs        = np.concatenate(all_obs, axis=0)
        gs         = np.concatenate(all_gs,  axis=0)
        actions    = np.concatenate(all_act, axis=0)
        advantages = np.concatenate(all_adv, axis=0)
        returns    = np.concatenate(all_ret, axis=0)
        log_probs  = np.concatenate(all_lp,  axis=0)

        # Normalize advantages (zero mean, unit std) — crucial for PPO stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "obs":        obs,
            "global_state": gs,
            "actions":    actions,
            "advantages": advantages,
            "returns":    returns,
            "old_log_probs": log_probs,
        }

    def make_minibatches(self, data: dict, minibatch_size: int):
        """
        Yield shuffled minibatches from collected data.
        """
        n       = len(data["obs"])
        indices = np.random.permutation(n)

        for start in range(0, n, minibatch_size):
            idx = indices[start: start + minibatch_size]
            yield {k: v[idx] for k, v in data.items()}
