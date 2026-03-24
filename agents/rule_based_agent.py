"""
agents/rule_based_agent.py

Python port of bidCalculator.js from the IPL Auction Simulator.
This is the handcoded baseline that MAPPO agents need to beat.
"""

import numpy as np
from typing import Dict, List, Optional

from environment.squad_validator import (
    can_bid_on_player, get_role_need_score, get_role_counts, SQUAD_RULES
)
from environment.team_config import TEAM_CONFIGS


def bid_increment(current_cr: float) -> float:
    if current_cr < 0.5:  return 0.05
    if current_cr < 2.0:  return 0.10
    if current_cr < 5.0:  return 0.25
    if current_cr < 10.0: return 0.50
    if current_cr < 20.0: return 0.25
    return 0.50


def scale_to_crore(score: float) -> float:
    if score <= 0:   return 0.0
    if score <= 10:  return score * 0.05
    if score <= 30:  return 0.5  + (score - 10)  * 0.15
    if score <= 60:  return 3.5  + (score - 30)  * 0.30
    if score <= 80:  return 12.5 + (score - 60)  * 0.50
    return min(30.0, 22.5 + (score - 80) * 0.50)


def get_age_factor(player: dict) -> float:
    age = player.get("age", 28)
    if age <= 22: return 1.25
    if age <= 25: return 1.15
    if age <= 28: return 1.05
    if age <= 31: return 0.95
    if age <= 34: return 0.82
    return 0.70


def get_archetype_score(player: dict, team_id: str) -> float:
    return float(TEAM_CONFIGS[team_id]["archetype_weights"].get(player.get("archetype", ""), 0))


def get_trait_score(player: dict, team_id: str) -> float:
    config = TEAM_CONFIGS[team_id]
    score  = 0.0
    for trait in player.get("traits", []):
        if trait in config["preferred_traits"]: score += 5.0
        if trait in config["avoided_traits"]:   score -= 5.0
    return max(-15.0, min(20.0, score))


def get_star_factor(player: dict, team_id: str) -> float:
    stars = player.get("star_rating", 1)
    if stars <= 2: return 0.0
    return (stars - 2) * 3.0 * TEAM_CONFIGS[team_id].get("marquee_bonus_multi", 1.0)


def get_overseas_impact_bonus(player: dict) -> float:
    if player.get("nationality") != "Overseas": return 0.0
    return max(0.0, (player.get("overseas_impact", 5) - 5) * 0.5)


def get_desperation_multiplier(player, squad, team_id, remaining_role_counts):
    role    = player.get("role", "")
    targets = TEAM_CONFIGS[team_id]["squad_targets"].get(role)
    if not targets: return 1.0
    current      = get_role_counts(squad).get(role, 0)
    remaining    = remaining_role_counts.get(role, 0)
    slots_needed = max(0, targets["min"] - current)
    if slots_needed == 0:             return 1.0
    if remaining == 0:                return 2.5
    if remaining <= slots_needed:     return 2.0
    if remaining <= slots_needed * 2: return 1.5
    if remaining <= slots_needed * 3: return 1.2
    return 1.0


def get_budget_modifier(budget_remaining, squad, team_id, total_players_needed):
    config = TEAM_CONFIGS[team_id]
    if total_players_needed <= 0: return 0.3
    reserved  = total_players_needed * 0.5
    spendable = max(0.0, budget_remaining - reserved)
    if spendable <= 0: return 0.25
    squad_fill = min(1.0, len(squad) / SQUAD_RULES["max_size"])
    divisor    = 4.0 - squad_fill * 2.0
    per_player = spendable / max(total_players_needed, 1)
    base       = max(0.4, min(1.3, per_player / divisor))
    agr = config["budget_aggressiveness"]
    if agr == "aggressive":   base = min(1.3, base * 1.12)
    if agr == "conservative": base = max(0.4, base * 0.88)
    return base


def get_overseas_multiplier(player, squad):
    if player.get("nationality") != "Overseas": return 1.0
    current = sum(1 for p in squad if p.get("nationality") == "Overseas")
    if current >= SQUAD_RULES["max_overseas"]: return 0.0
    return 1.0 - (current / SQUAD_RULES["max_overseas"]) * 0.3


def get_total_players_needed(squad, team_id):
    targets     = TEAM_CONFIGS[team_id]["squad_targets"]
    role_counts = get_role_counts(squad)
    needed      = sum(max(0, l["min"] - role_counts.get(r, 0)) for r, l in targets.items())
    return max(needed, max(0, SQUAD_RULES["min_size"] - len(squad)))


SCARE_CHANCE = {"very_low":0.03,"low":0.07,"medium":0.13,"high":0.22,"very_high":0.35}
SCARE_MULTI  = {"very_low":(1.5,2.0),"low":(1.5,2.5),"medium":(2.0,3.5),
                "high":(2.5,5.0),"very_high":(3.0,8.0)}


class RuleBasedAgent:
    """
    Handcoded bidding agent — port of JS bidCalculator.
    Computes a value ceiling per player based on role need,
    archetype fit, star factor, budget state, and team personality.
    """

    def __init__(self, team_id: str, rng=None):
        self.team_id = team_id
        self.config  = TEAM_CONFIGS[team_id]
        self.rng     = rng or np.random.default_rng()

    def compute_value(self, player, squad, budget, remaining_role_counts,
                      current_bid, leading_team):
        if not can_bid_on_player(squad, player):
            return {"max_bid": 0.0, "will_bid": False, "is_scare": False, "actual_bid": 0.0}
        if budget <= current_bid:
            return {"max_bid": 0.0, "will_bid": False, "is_scare": False, "actual_bid": 0.0}

        total_needed = get_total_players_needed(squad, self.team_id)
        role_need    = get_role_need_score(squad, player, self.config["squad_targets"], remaining_role_counts)
        arch_score   = get_archetype_score(player, self.team_id)
        trait_score  = get_trait_score(player, self.team_id)
        star_factor  = get_star_factor(player, self.team_id)
        os_bonus     = get_overseas_impact_bonus(player)
        age_factor   = get_age_factor(player)
        despe_multi  = get_desperation_multiplier(player, squad, self.team_id, remaining_role_counts)
        budget_multi = get_budget_modifier(budget, squad, self.team_id, total_needed)
        os_multi     = get_overseas_multiplier(player, squad)

        raw        = (role_need + arch_score + trait_score + star_factor + os_bonus) \
                     * despe_multi * budget_multi * os_multi * age_factor
        comfort_cr = min(scale_to_crore(raw), self.config["value_ceiling_cr"], budget)
        comfort_cr = round(comfort_cr, 2)

        sf = 1.30 if self.config["budget_aggressiveness"]=="aggressive" else \
             1.10 if self.config["budget_aggressiveness"]=="conservative" else 1.18
        crit  = 1.10 if despe_multi >= 2.0 else 1.0
        fill  = min(1.0, len(squad) / SQUAD_RULES["max_size"])
        cap_f = 0.35 + fill * 0.25
        sp    = max(0.0, budget - max(0, total_needed-1)*0.5)
        hard  = sp * cap_f
        stretch = round(min(comfort_cr*sf*crit, self.config["value_ceiling_cr"]*1.1, hard, budget), 2)
        stretch = max(stretch, comfort_cr)

        # Rivalry bump
        if (leading_team and leading_team != self.team_id
                and leading_team in self.config["rival_teams"]
                and player.get("star_rating", 1) >= 3
                and self.rng.random() < 0.20):
            stretch = min(stretch * 1.10, budget)

        next_bid = round(current_bid + bid_increment(current_bid), 2)
        if next_bid > stretch:
            return {"max_bid": stretch, "will_bid": False, "is_scare": False, "actual_bid": 0.0}

        scare_chance = SCARE_CHANCE.get(self.config["scare_bid_tendency"], 0.05)
        is_scare = (self.rng.random() < scare_chance and current_bid > 0
                    and leading_team != self.team_id)
        actual = next_bid
        if is_scare:
            mn, mx = SCARE_MULTI.get(self.config["scare_bid_tendency"], (1.5, 2.5))
            jump   = round(current_bid * (mn + self.rng.random()*(mx-mn)), 2)
            if next_bid < jump <= stretch <= budget:
                actual = jump

        return {"max_bid": stretch, "will_bid": True, "is_scare": is_scare, "actual_bid": actual}

    def act(self, player, squad, budget, remaining_role_counts, current_bid, leading_team):
        """Returns 1 (BID) or 0 (PASS)."""
        if leading_team == self.team_id:
            return 0
        result = self.compute_value(player, squad, budget, remaining_role_counts,
                                     current_bid, leading_team)
        return 1 if result["will_bid"] else 0

    def reset(self):
        pass
