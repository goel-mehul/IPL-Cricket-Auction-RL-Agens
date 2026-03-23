"""
tests/test_environment.py

Unit tests for the IPL auction environment.
Run with: pytest tests/ -v
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.auction_env import IPLAuctionEnv, OBS_DIM, GLOBAL_STATE_DIM
from environment.squad_validator import (
    validate_squad, can_bid_on_player, get_role_need_score,
    get_role_counts, compute_final_score, SQUAD_RULES
)
from environment.pool_generator import generate_pools, get_remaining_role_counts
from environment.team_config import TEAM_CONFIGS, ALL_TEAM_IDS, RETENTION_SLABS


# ── squad_validator tests ─────────────────────────────────────────

class TestSquadValidator:

    def _make_player(self, role, nationality="Indian", overall=75, pid=None):
        return {
            "id": pid or np.random.randint(1000, 9999),
            "name": f"Player_{role}",
            "role": role,
            "nationality": nationality,
            "overall": overall,
            "star_rating": 3,
            "traits": [],
            "archetype": "Anchor",
        }

    def test_validate_empty_squad(self):
        targets = TEAM_CONFIGS["MI"]["squad_targets"]
        v = validate_squad([], targets)
        assert v.squad_size == 0
        assert not v.all_mins_met
        assert v.needs_more_players

    def test_validate_meets_minimums(self):
        targets = TEAM_CONFIGS["CSK"]["squad_targets"]
        squad = (
            [self._make_player("BAT",    pid=i)    for i in range(4)] +
            [self._make_player("WK-BAT", pid=10+i) for i in range(1)] +
            [self._make_player("ALL",    pid=20+i) for i in range(2)] +
            [self._make_player("PACE",   pid=30+i) for i in range(4)] +
            [self._make_player("SPIN",   pid=40+i) for i in range(2)]
        )
        v = validate_squad(squad, targets)
        assert v.all_mins_met

    def test_overseas_limit(self):
        targets = TEAM_CONFIGS["MI"]["squad_targets"]
        squad = [self._make_player("BAT", nationality="Overseas", pid=i) for i in range(6)]
        v = validate_squad(squad, targets)
        assert not v.overseas_ok

    def test_can_bid_squad_full(self):
        squad = [self._make_player("BAT", pid=i) for i in range(SQUAD_RULES["max_size"])]
        player = self._make_player("PACE", pid=9999)
        assert not can_bid_on_player(squad, player)

    def test_can_bid_overseas_limit(self):
        squad = [
            self._make_player("BAT", nationality="Overseas", pid=i)
            for i in range(SQUAD_RULES["max_overseas"])
        ]
        overseas_player = self._make_player("PACE", nationality="Overseas", pid=9999)
        assert not can_bid_on_player(squad, overseas_player)
        indian_player = self._make_player("PACE", nationality="Indian", pid=8888)
        assert can_bid_on_player(squad, indian_player)

    def test_role_need_score_critical(self):
        """Role below minimum → high need score."""
        targets = TEAM_CONFIGS["SRH"]["squad_targets"]  # PACE min=6
        squad  = []  # no pace bowlers
        player = self._make_player("PACE")
        remaining = {"PACE": 20, "BAT": 30, "WK-BAT": 5, "ALL": 10, "SPIN": 8}
        score = get_role_need_score(squad, player, targets, remaining)
        assert score >= 25.0  # critical need

    def test_role_need_score_at_max(self):
        """Role at max → zero need."""
        targets = TEAM_CONFIGS["MI"]["squad_targets"]  # BAT max=7
        squad   = [self._make_player("BAT", pid=i) for i in range(7)]
        player  = self._make_player("BAT")
        remaining = {"BAT": 10, "WK-BAT": 5, "ALL": 8, "PACE": 15, "SPIN": 6}
        score = get_role_need_score(squad, player, targets, remaining)
        assert score == 0.0

    def test_compute_final_score_empty(self):
        result = compute_final_score([], TEAM_CONFIGS["MI"]["squad_targets"],
                                     TEAM_CONFIGS["MI"]["playing_xi"])
        assert result["final_score"] == 0

    def test_compute_final_score_full_squad(self):
        from environment.auction_env import load_players
        players = load_players()
        # Take first 17 players as a "squad"
        squad  = players[:17]
        result = compute_final_score(squad, TEAM_CONFIGS["MI"]["squad_targets"],
                                     TEAM_CONFIGS["MI"]["playing_xi"])
        assert 0 <= result["final_score"] <= 100
        assert result["grade"] in ["S", "A", "B", "C", "D"]
        assert result["squad_size"] == 17


# ── pool_generator tests ──────────────────────────────────────────

class TestPoolGenerator:

    def test_pools_generated(self):
        from environment.auction_env import load_players
        players = load_players()
        pools   = generate_pools(players, seed=0)
        assert len(pools) > 0
        # Marquee pool first
        assert pools[0]["id"] == "marquee"

    def test_retained_excluded(self):
        from environment.auction_env import load_players
        players     = load_players()
        retained_id = {players[0]["id"]}
        pools       = generate_pools(players, retained_ids=retained_id, seed=0)
        all_ids     = {p["id"] for pool in pools for p in pool["players"]}
        assert players[0]["id"] not in all_ids

    def test_no_duplicates(self):
        from environment.auction_env import load_players
        players = load_players()
        pools   = generate_pools(players, seed=0)
        all_ids = [p["id"] for pool in pools for p in pool["players"]]
        assert len(all_ids) == len(set(all_ids)), "Duplicate players in pools"

    def test_all_players_present(self):
        from environment.auction_env import load_players
        players = load_players()
        pools   = generate_pools(players, seed=0)
        all_ids = {p["id"] for pool in pools for p in pool["players"]}
        orig_ids = {p["id"] for p in players}
        assert all_ids == orig_ids

    def test_determinism(self):
        from environment.auction_env import load_players
        players = load_players()
        pools1  = generate_pools(players, seed=42)
        pools2  = generate_pools(players, seed=42)
        names1  = [p["name"] for p in pools1[0]["players"]]
        names2  = [p["name"] for p in pools2[0]["players"]]
        assert names1 == names2

    def test_different_seeds_differ(self):
        from environment.auction_env import load_players
        players = load_players()
        pools1  = generate_pools(players, seed=1)
        pools2  = generate_pools(players, seed=2)
        names1  = [p["name"] for p in pools1[0]["players"]]
        names2  = [p["name"] for p in pools2[0]["players"]]
        assert names1 != names2


# ── auction_env tests ─────────────────────────────────────────────

class TestAuctionEnv:

    def setup_method(self):
        self.env = IPLAuctionEnv(seed=42)

    def test_reset_returns_obs(self):
        obs, infos = self.env.reset(seed=0)
        assert set(obs.keys()) == set(ALL_TEAM_IDS)
        for agent, o in obs.items():
            assert o.shape == (OBS_DIM,)
            assert o.dtype == np.float32

    def test_obs_bounds(self):
        obs, _ = self.env.reset(seed=0)
        for agent, o in obs.items():
            assert o.min() >= 0.0, f"{agent} obs min < 0"
            assert o.max() <= 1.0 + 1e-5, f"{agent} obs max > 1"

    def test_global_state_shape(self):
        self.env.reset(seed=0)
        state = self.env.state()
        assert state.shape == (OBS_DIM * len(ALL_TEAM_IDS),)

    def test_agent_selection_cycles(self):
        self.env.reset(seed=0)
        seen = set()
        for _ in range(len(ALL_TEAM_IDS)):
            agent = self.env.agent_selection
            seen.add(agent)
            obs, r, term, trunc, info = self.env.last()
            self.env.step(0)
        assert seen == set(ALL_TEAM_IDS)

    def test_episode_terminates(self):
        """Full episode with random actions must terminate."""
        np.random.seed(0)
        self.env.reset(seed=0)
        step_count = 0
        max_steps  = 30000

        while self.env.agents and step_count < max_steps:
            obs, r, term, trunc, _ = self.env.last()
            if term or trunc:
                self.env.step(None)
            else:
                self.env.step(1 if np.random.random() < 0.3 else 0)
            step_count += 1

        assert step_count < max_steps, "Episode did not terminate"
        assert len(self.env.agents) == 0
        assert self.env.get_state_dict()["phase"] == "ended"

    def test_no_player_in_multiple_squads(self):
        """No player should appear in more than one team's squad."""
        np.random.seed(1)
        self.env.reset(seed=1)

        while self.env.agents:
            obs, r, term, trunc, _ = self.env.last()
            if term or trunc:
                self.env.step(None)
            else:
                self.env.step(1 if np.random.random() < 0.3 else 0)

        state = self.env.get_state_dict()
        all_player_ids = []
        for team_id in ALL_TEAM_IDS:
            for p in state["teams"][team_id]["squad"]:
                all_player_ids.append(p["id"])

        assert len(all_player_ids) == len(set(all_player_ids)), \
            "Same player in multiple squads!"

    def test_budgets_non_negative(self):
        """No team should have negative budget."""
        np.random.seed(2)
        self.env.reset(seed=2)

        while self.env.agents:
            obs, r, term, trunc, _ = self.env.last()
            if term or trunc:
                self.env.step(None)
            else:
                self.env.step(1 if np.random.random() < 0.3 else 0)

        state = self.env.get_state_dict()
        for team_id in ALL_TEAM_IDS:
            budget = state["teams"][team_id]["budget"]
            assert budget >= -0.01, f"{team_id} has negative budget: {budget}"

    def test_retention_excluded_from_pool(self):
        """Retained players must not appear in auction pools."""
        self.env.reset(seed=0)
        state = self.env.get_state_dict()
        retained_names = set()
        for team_id in ALL_TEAM_IDS:
            for p in state["teams"][team_id]["squad"]:
                if p.get("is_retained"):
                    retained_names.add(p["name"])

        pool_names = {
            p["name"]
            for pool in state["pools"]
            for p in pool["players"]
        }
        overlap = retained_names & pool_names
        assert len(overlap) == 0, f"Retained players in pool: {overlap}"

    def test_retentions_reduce_budget(self):
        """Teams with retentions should start with less than 100 Cr."""
        self.env.reset(seed=0)
        state = self.env.get_state_dict()
        for team_id in ALL_TEAM_IDS:
            team   = state["teams"][team_id]
            config = TEAM_CONFIGS[team_id]
            num_ret = len([p for p in team["squad"] if p.get("is_retained")])
            expected_budget = 100.0 - sum(RETENTION_SLABS[:num_ret])
            assert abs(team["budget"] - expected_budget) < 0.01, \
                f"{team_id}: expected budget {expected_budget}, got {team['budget']}"

    def test_final_scores_returned(self):
        """After episode, get_final_scores returns dict for all teams."""
        np.random.seed(3)
        self.env.reset(seed=3)
        while self.env.agents:
            obs, r, term, trunc, _ = self.env.last()
            if term or trunc:
                self.env.step(None)
            else:
                self.env.step(1 if np.random.random() < 0.3 else 0)

        scores = self.env.get_final_scores()
        assert set(scores.keys()) == set(ALL_TEAM_IDS)
        for tid, sc in scores.items():
            assert "final_score" in sc
            assert "grade" in sc
            assert 0 <= sc["final_score"] <= 100
