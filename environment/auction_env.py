"""
auction_env.py

IPL Auction as a PettingZoo AEC (Agent Environment Cycle) environment.

Each step: the current team decides BID (1) or PASS (0) on the
current player. The highest bidder after the timer wins the player.

Observation space (per agent, dim=65):
  [0]     player_overall_norm       — overall / 100
  [1-5]   player_role_onehot        — BAT/WK/ALL/PACE/SPIN
  [6]     player_nationality        — 0=Indian, 1=Overseas
  [7]     player_base_price_norm    — base_price / 2.0
  [8]     player_star_norm          — star_rating / 5.0
  [9]     current_bid_norm          — current_bid / budget
  [10]    is_leading                — 1 if this team is leading
  [11-15] my_role_counts_norm       — per-role count / 8
  [16]    my_overseas_norm          — overseas_count / 5
  [17]    my_squad_size_norm        — squad_size / 19
  [18]    my_budget_norm            — budget / 100
  [19]    my_slots_remaining_norm   — (max_size - squad_size) / 19
  [20-24] role_need_scores_norm     — getRoleNeedScore / 40
  [25-29] remaining_pool_norm       — remaining per role / 50
  [30]    remaining_overseas_norm   — remaining overseas / 30
  [31]    auction_progress          — pool_idx / total_pools
  [32]    budget_per_slot_norm      — budget / max(slots, 1) / 10
  [33-42] opponent_budgets_norm     — 9 opponents' budgets / 100
  [43-52] opponent_squad_sizes_norm — 9 opponents' squad sizes / 19
  [53-57] opponent_overseas_norm    — 9 opponents' overseas / 5
  [58-62] my_role_deficits_norm     — slots_needed per role / 3
  [63]    player_archetype_norm     — archetypeWeight / 25
  [64]    player_trait_score_norm   — trait match score / 20

Action space: Discrete(2) — 0=PASS, 1=BID

Global state (for centralised critic, dim=200):
  All agents' observations concatenated (10 × 65 = 650, projected to 200)
  Actually we pass raw 10×65 and let the critic network handle it.
"""

import json
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path

try:
    from pettingzoo import AECEnv
    from pettingzoo.utils import agent_selector, wrappers
    from gymnasium import spaces
    PETTINGZOO_AVAILABLE = True
except ImportError:
    # Fallback: define minimal base class so env works without pettingzoo installed
    PETTINGZOO_AVAILABLE = False
    class AECEnv:
        pass
    class spaces:
        @staticmethod
        def Discrete(n): return n
        @staticmethod
        def Box(low, high, shape, dtype): return shape

from environment.team_config import TEAM_CONFIGS, ALL_TEAM_IDS, RETENTION_SLABS
from environment.squad_validator import (
    validate_squad, can_bid_on_player, get_role_need_score,
    get_role_counts, compute_final_score, SQUAD_RULES, ROLES
)
from environment.pool_generator import (
    generate_pools, get_remaining_role_counts, flatten_pools
)

# ── Constants ─────────────────────────────────────────────────────
OBS_DIM          = 65
GLOBAL_STATE_DIM = 10 * OBS_DIM   # full concat of all agents' obs

DATA_PATH = Path(__file__).parent.parent / "data" / "players.json"


def load_players() -> List[dict]:
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("players", [])


def _bid_increment(current_cr: float) -> float:
    if current_cr < 0.5:  return 0.05
    if current_cr < 2.0:  return 0.10
    if current_cr < 5.0:  return 0.25
    if current_cr < 10.0: return 0.50
    if current_cr < 20.0: return 0.25
    return 0.50


class IPLAuctionEnv(AECEnv):
    """
    IPL Auction multi-agent environment.

    AEC cycle: agents take turns bidding on the current player.
    One 'round' = all 10 agents get one BID/PASS decision.
    After `timer_steps` rounds with no new bid, player is sold/unsold.

    Usage:
        env = IPLAuctionEnv(seed=42)
        env.reset()
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                action = None
            else:
                action = policy(obs)
            env.step(action)
    """

    metadata = {"name": "ipl_auction_v1", "render_modes": []}

    def __init__(
        self,
        retentions_enabled: bool = True,
        seed: Optional[int] = None,
        reward_config: Optional[dict] = None,
    ):
        super().__init__()

        self.all_players      = load_players()
        self.retentions_enabled = retentions_enabled
        self._seed            = seed
        self.reward_config    = reward_config or {
            "final_score_weight":  1.0,
            "completeness_bonus":  0.3,
            "budget_waste_penalty": 0.05,
            "overseas_bonus":      0.1,
        }

        # PettingZoo required attributes
        self.agents         = list(ALL_TEAM_IDS)
        self.possible_agents = list(ALL_TEAM_IDS)

        if PETTINGZOO_AVAILABLE:
            self.observation_spaces = {
                a: spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
                for a in self.agents
            }
            self.action_spaces = {
                a: spaces.Discrete(2)
                for a in self.agents
            }

        # Internal state (populated in reset())
        self._state: Optional[dict] = None

    # ── PettingZoo API ─────────────────────────────────────────────

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed

        self.agents = list(ALL_TEAM_IDS)

        # Build retention data
        retained_ids = set()
        teams = {}
        for team_id in ALL_TEAM_IDS:
            config = TEAM_CONFIGS[team_id]
            if self.retentions_enabled:
                ret_names  = set(config["retentions"])
                retained   = [p for p in self.all_players if p["name"] in ret_names]
            else:
                retained = []
            cost   = sum(RETENTION_SLABS[:len(retained)])
            budget = 100.0 - cost
            for p in retained:
                retained_ids.add(p["id"])
            teams[team_id] = {
                "budget": budget,
                "squad":  [dict(p, sold_price=RETENTION_SLABS[i], is_retained=True)
                           for i, p in enumerate(retained)],
                "done":   False,
            }

        pools = generate_pools(self.all_players, retained_ids, seed=self._seed)

        self._state = {
            "teams":               teams,
            "pools":               pools,
            "pool_idx":            0,
            "player_idx":          0,
            "current_bid_cr":      self._current_player(pools, 0, 0)["base_price"] if pools else 0,
            "leading_team":        None,
            "no_bid_rounds":       0,       # rounds with zero new bids
            "timer_max":           2,       # hammer after 2 no-bid rounds
            "had_bid_this_round":  False,   # did anyone bid this round?
            "phase":               "auction",
            "unsold_players":      [],
            "pass_flags":          {t: False for t in ALL_TEAM_IDS},
        }

        # AEC bookkeeping
        self._agent_selector   = _AgentSelector(self.agents)
        self.agent_selection   = self._agent_selector.reset()
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.rewards           = {a: 0.0 for a in self.agents}
        self.terminations      = {a: False for a in self.agents}
        self.truncations       = {a: False for a in self.agents}
        self.infos             = {a: {} for a in self.agents}
        self._observations     = {a: self._get_obs(a) for a in self.agents}

        return self._observations, self.infos

    def step(self, action):
        if not self.agents:
            return

        agent = self.agent_selection
        s     = self._state

        if self.terminations[agent] or self.truncations[agent]:
            # Skip dead agents — just advance selector
            self.agent_selection = self._agent_selector.next()
            return

        # ── Process action ────────────────────────────────────────
        player = self._current_player(s["pools"], s["pool_idx"], s["player_idx"])
        team   = s["teams"][agent]

        if action == 1 and player:   # BID
            next_bid = s["current_bid_cr"] + _bid_increment(s["current_bid_cr"])
            next_bid = round(next_bid, 2)

            can_bid  = (
                can_bid_on_player(team["squad"], player)
                and team["budget"] >= next_bid
                and s["leading_team"] != agent
            )
            if can_bid:
                s["current_bid_cr"]      = next_bid
                s["leading_team"]        = agent
                s["no_bid_rounds"]       = 0      # reset no-bid counter
                s["had_bid_this_round"]  = True
                s["pass_flags"][agent]   = False

        else:   # PASS
            s["pass_flags"][agent] = True

        # ── Advance AEC ───────────────────────────────────────────
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards[agent] = 0.0

        # Check if this is the LAST agent in the round BEFORE advancing
        round_complete = self._agent_selector.is_last()

        self.agent_selection = self._agent_selector.next()

        # ── Resolve round after all agents have acted ─────────────
        if round_complete:
            self._resolve_round()

        self._observations = {a: self._get_obs(a) for a in self.agents}

    def observe(self, agent):
        return self._observations.get(agent, np.zeros(OBS_DIM, dtype=np.float32))

    def last(self, observe=True):
        agent = self.agent_selection
        obs   = self.observe(agent) if observe else None
        return (
            obs,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def state(self) -> np.ndarray:
        """Global state for centralised critic — concat all obs."""
        obs_list = [self._get_obs(a) for a in ALL_TEAM_IDS]
        return np.concatenate(obs_list, axis=0).astype(np.float32)

    # ── Internal logic ─────────────────────────────────────────────

    def _resolve_round(self):
        """Called after all 10 agents have acted in a round."""
        s = self._state

        player = self._current_player(s["pools"], s["pool_idx"], s["player_idx"])
        if not player:
            self._end_auction()
            return

        if not s["had_bid_this_round"]:
            s["no_bid_rounds"] += 1
        # Always reset round flag
        s["had_bid_this_round"] = False

        hammer = s["no_bid_rounds"] >= s["timer_max"]

        if hammer:
            if s["leading_team"]:
                self._sell_player(player, s["leading_team"], s["current_bid_cr"])
            else:
                self._unsold_player(player)
            self._advance_to_next_player()

        # Reset pass flags for next round
        s["pass_flags"] = {t: False for t in ALL_TEAM_IDS}

    def _sell_player(self, player: dict, team_id: str, price: float):
        s = self._state
        team = s["teams"][team_id]
        sold = dict(player, sold_price=price, is_retained=False)
        team["squad"].append(sold)
        team["budget"] = round(team["budget"] - price, 2)

        # Intermediate completeness reward
        rc = self.reward_config
        prev_val = validate_squad(
            team["squad"][:-1], TEAM_CONFIGS[team_id]["squad_targets"]
        )
        new_val  = validate_squad(
            team["squad"], TEAM_CONFIGS[team_id]["squad_targets"]
        )
        if not prev_val.all_mins_met and new_val.all_mins_met:
            self.rewards[team_id] += rc["completeness_bonus"]
            self._cumulative_rewards[team_id] += rc["completeness_bonus"]

    def _unsold_player(self, player: dict):
        s = self._state
        if s["phase"] == "unsold":
            return   # skip if already halved price and still unsold
        s["unsold_players"].append(player)

    def _advance_to_next_player(self):
        s = self._state
        s["player_idx"]       += 1
        s["leading_team"]      = None
        s["no_bid_rounds"]     = 0
        s["had_bid_this_round"] = False

        pool = s["pools"][s["pool_idx"]] if s["pool_idx"] < len(s["pools"]) else None
        if pool and s["player_idx"] >= len(pool["players"]):
            s["pool_idx"]   += 1
            s["player_idx"] = 0

        if s["pool_idx"] >= len(s["pools"]):
            if s["phase"] == "auction" and s["unsold_players"]:
                self._start_unsold_round()
            else:
                self._end_auction()
            return

        player = self._current_player(s["pools"], s["pool_idx"], s["player_idx"])
        if player:
            s["current_bid_cr"] = player["base_price"]
        else:
            self._end_auction()

    def _start_unsold_round(self):
        s = self._state
        s["phase"] = "unsold"
        # Halve base prices, put back in pool as a single "unsold" pool
        unsold_halved = []
        for p in s["unsold_players"]:
            p2 = dict(p, base_price=round(p["base_price"] * 0.5, 2))
            unsold_halved.append(p2)
        s["pools"].append({
            "id":         "unsold_round",
            "label":      "Unsold Round",
            "role":       None,
            "set_number": 1,
            "players":    unsold_halved,
        })
        s["unsold_players"] = []
        s["pool_idx"]       = len(s["pools"]) - 1
        s["player_idx"]     = 0
        player = unsold_halved[0] if unsold_halved else None
        if player:
            s["current_bid_cr"] = player["base_price"]
        else:
            self._end_auction()

    def _end_auction(self):
        s = self._state
        s["phase"] = "ended"

        # Compute final rewards for all agents
        rc = self.reward_config
        for team_id in ALL_TEAM_IDS:
            team   = s["teams"][team_id]
            config = TEAM_CONFIGS[team_id]
            result = compute_final_score(
                team["squad"],
                config["squad_targets"],
                config["playing_xi"],
            )
            # Normalise final score to roughly 0-1
            score_norm = result["final_score"] / 100.0

            # Budget waste penalty
            waste_penalty = 0.0
            if team["budget"] > 30:
                waste_penalty = rc["budget_waste_penalty"] * (team["budget"] - 30) / 70

            # Overseas bonus
            overseas_bonus = rc["overseas_bonus"] * (result["overseas_util"] / 100.0)

            final_reward = (
                score_norm * rc["final_score_weight"]
                - waste_penalty
                + overseas_bonus
            )

            self.rewards[team_id]              = final_reward
            self._cumulative_rewards[team_id] += final_reward
            self.infos[team_id]  = result
            self.terminations[team_id] = True

        self.agents = []

    # ── Observation builder ────────────────────────────────────────

    def _get_obs(self, agent: str) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        s   = self._state
        if s is None:
            return obs

        team   = s["teams"][agent]
        config = TEAM_CONFIGS[agent]
        player = self._current_player(s["pools"], s["pool_idx"], s["player_idx"])

        if player is None:
            return obs

        # ── Player features [0-9] ─────────────────────────────────
        obs[0] = player.get("overall", 70) / 100.0
        role   = player.get("role", "BAT")
        for i, r in enumerate(ROLES):
            obs[1 + i] = 1.0 if role == r else 0.0
        obs[6] = 1.0 if player.get("nationality") == "Overseas" else 0.0
        obs[7] = min(1.0, player.get("base_price", 0.5) / 2.0)
        obs[8] = player.get("star_rating", 1) / 5.0

        # ── Bid context [9-10] ─────────────────────────────────────
        budget = max(team["budget"], 0.01)
        obs[9]  = min(1.0, s["current_bid_cr"] / budget)
        obs[10] = 1.0 if s["leading_team"] == agent else 0.0

        # ── My squad state [11-19] ────────────────────────────────
        rc_my = get_role_counts(team["squad"])
        for i, r in enumerate(ROLES):
            obs[11 + i] = rc_my.get(r, 0) / 8.0
        overseas = sum(1 for p in team["squad"] if p.get("nationality") == "Overseas")
        obs[16] = overseas / 5.0
        obs[17] = len(team["squad"]) / SQUAD_RULES["max_size"]
        obs[18] = team["budget"] / 100.0
        obs[19] = max(0, SQUAD_RULES["max_size"] - len(team["squad"])) / SQUAD_RULES["max_size"]

        # ── Role need scores [20-24] ──────────────────────────────
        remaining = get_remaining_role_counts(s["pools"], s["pool_idx"], s["player_idx"])
        for i, r in enumerate(ROLES):
            tmp_player = {"role": r, "nationality": "Indian"}
            obs[20 + i] = get_role_need_score(
                team["squad"], tmp_player, config["squad_targets"], remaining
            ) / 40.0

        # ── Remaining pool [25-30] ────────────────────────────────
        for i, r in enumerate(ROLES):
            obs[25 + i] = remaining.get(r, 0) / 50.0
        obs[30] = sum(
            1 for pool in s["pools"][s["pool_idx"]:]
            for p in pool["players"]
            if p.get("nationality") == "Overseas"
        ) / 30.0

        # ── Auction progress [31-32] ──────────────────────────────
        total_pools = len(s["pools"])
        obs[31] = s["pool_idx"] / max(total_pools, 1)
        slots_left = max(SQUAD_RULES["max_size"] - len(team["squad"]), 1)
        obs[32] = min(1.0, team["budget"] / slots_left / 10.0)

        # ── Opponent states [33-57] ───────────────────────────────
        opponents = [t for t in ALL_TEAM_IDS if t != agent]
        for i, opp in enumerate(opponents[:9]):
            opp_team = s["teams"][opp]
            obs[33 + i] = opp_team["budget"] / 100.0
            obs[43 + i] = len(opp_team["squad"]) / SQUAD_RULES["max_size"]
            opp_os = sum(1 for p in opp_team["squad"] if p.get("nationality") == "Overseas")
            obs[53 + i] = opp_os / 5.0

        # ── Role deficits [58-62] ─────────────────────────────────
        targets = config["squad_targets"]
        for i, r in enumerate(ROLES):
            needed = max(0, targets[r]["min"] - rc_my.get(r, 0))
            obs[58 + i] = needed / 3.0

        # ── Player fit [63-64] ────────────────────────────────────
        archetype_w = config["archetype_weights"].get(player.get("archetype", ""), 0)
        obs[63] = archetype_w / 25.0
        traits = player.get("traits", [])
        trait_score = sum(5 for t in traits if t in config["preferred_traits"])
        trait_score -= sum(5 for t in traits if t in config["avoided_traits"])
        obs[64] = min(1.0, max(0.0, (trait_score + 20) / 40.0))

        return np.clip(obs, 0.0, 1.0)

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _current_player(pools, pool_idx, player_idx):
        if pool_idx >= len(pools):
            return None
        pool = pools[pool_idx]
        if player_idx >= len(pool["players"]):
            return None
        return pool["players"][player_idx]

    def get_state_dict(self) -> dict:
        """Return full state for debugging / logging."""
        return self._state

    def get_final_scores(self) -> Dict[str, dict]:
        """After episode ends, return final scores for all teams."""
        if self._state["phase"] != "ended":
            return {}
        scores = {}
        for team_id in ALL_TEAM_IDS:
            team   = self._state["teams"][team_id]
            config = TEAM_CONFIGS[team_id]
            scores[team_id] = compute_final_score(
                team["squad"], config["squad_targets"], config["playing_xi"]
            )
        return scores


# ── Minimal agent selector (replaces pettingzoo's if not installed) ──

class _AgentSelector:
    """Cycles through agents in order."""
    def __init__(self, agents):
        self._agents = list(agents)
        self._idx    = 0

    def reset(self):
        self._idx = 0
        return self._agents[0]

    def next(self):
        self._idx = (self._idx + 1) % len(self._agents)
        return self._agents[self._idx]

    def is_last(self):
        """True when the current agent is the last in the cycle."""
        return self._idx == len(self._agents) - 1
