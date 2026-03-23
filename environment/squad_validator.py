"""
squad_validator.py

Port of squadValidator.js — pure logic, no UI.
Validates squad composition and computes role need scores
used by both the environment and the rule-based agent.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# ── Squad Rules ───────────────────────────────────────────────────
SQUAD_RULES = {
    "min_size":     15,
    "max_size":     19,
    "max_overseas": 5,
}

ROLES = ["BAT", "WK-BAT", "ALL", "PACE", "SPIN"]


@dataclass
class RoleStatus:
    count:      int
    min:        int
    max:        int
    ideal:      int
    below_min:  bool
    at_ideal:   bool
    above_max:  bool
    slots_needed:    int
    slots_remaining: int


@dataclass
class SquadValidation:
    valid:               bool
    squad_size:          int
    overseas_count:      int
    overseas_slots_left: int
    overseas_ok:         bool
    size_ok:             bool
    all_mins_met:        bool
    all_maxs_respected:  bool
    can_add_more:        bool
    needs_more_players:  bool
    role_status:         Dict[str, RoleStatus]


def get_role_counts(squad: List[dict]) -> Dict[str, int]:
    counts = {r: 0 for r in ROLES}
    for p in squad:
        role = p.get("role", "")
        if role in counts:
            counts[role] += 1
    return counts


def validate_squad(squad: List[dict], squad_targets: Dict) -> SquadValidation:
    """
    Returns full validation snapshot of a squad against its targets.
    squad_targets: { role: { min, max, ideal } }
    """
    role_counts  = get_role_counts(squad)
    overseas     = sum(1 for p in squad if p.get("nationality") == "Overseas")
    squad_size   = len(squad)

    role_status  = {}
    all_mins_met = True
    all_maxs_ok  = True

    for role, limits in squad_targets.items():
        count     = role_counts.get(role, 0)
        below_min = count < limits["min"]
        above_max = count > limits["max"]
        if below_min: all_mins_met = False
        if above_max: all_maxs_ok  = False

        role_status[role] = RoleStatus(
            count=count,
            min=limits["min"],
            max=limits["max"],
            ideal=limits["ideal"],
            below_min=below_min,
            at_ideal=(count == limits["ideal"]),
            above_max=above_max,
            slots_needed=max(0, limits["min"] - count),
            slots_remaining=max(0, limits["max"] - count),
        )

    overseas_ok = overseas <= SQUAD_RULES["max_overseas"]
    size_ok     = SQUAD_RULES["min_size"] <= squad_size <= SQUAD_RULES["max_size"]

    return SquadValidation(
        valid=all_mins_met and all_maxs_ok and overseas_ok and size_ok,
        squad_size=squad_size,
        overseas_count=overseas,
        overseas_slots_left=SQUAD_RULES["max_overseas"] - overseas,
        overseas_ok=overseas_ok,
        size_ok=size_ok,
        all_mins_met=all_mins_met,
        all_maxs_respected=all_maxs_ok,
        can_add_more=squad_size < SQUAD_RULES["max_size"],
        needs_more_players=squad_size < SQUAD_RULES["min_size"],
        role_status=role_status,
    )


def can_bid_on_player(squad: List[dict], player: dict) -> bool:
    """Hard eligibility check — squad cap and overseas limit."""
    if len(squad) >= SQUAD_RULES["max_size"]:
        return False
    if player.get("nationality") == "Overseas":
        overseas = sum(1 for p in squad if p.get("nationality") == "Overseas")
        if overseas >= SQUAD_RULES["max_overseas"]:
            return False
    return True


def get_role_need_score(
    squad: List[dict],
    player: dict,
    squad_targets: Dict,
    remaining_role_counts: Dict[str, int],
) -> float:
    """
    How urgently does this team need this player's role?
    Returns 0–40. Used in bid value formula.
    """
    role   = player.get("role", "")
    limits = squad_targets.get(role)
    if not limits:
        return 0.0

    current      = get_role_counts(squad).get(role, 0)
    remaining    = remaining_role_counts.get(role, 0)
    slots_needed = max(0, limits["min"] - current)
    slots_ideal  = max(0, limits["ideal"] - current)

    if current >= limits["max"]:
        return 0.0  # already at max

    if slots_needed > 0:
        scarcity = 2.0 if remaining <= slots_needed else 1.2
        return min(40.0, 25.0 + slots_needed * 5.0 * scarcity)

    if slots_ideal > 0:
        scarcity = 1.5 if remaining <= 3 else 1.0
        return min(20.0, 10.0 + slots_ideal * 4.0 * scarcity)

    return 3.0  # at or above ideal — low desire


def should_declare_done(
    squad: List[dict],
    budget_remaining: float,
    squad_targets: Dict,
    min_player_price: float = 0.2,
) -> bool:
    """
    Should this team stop bidding in the unsold round?
    Never true if squad is below minimum size.
    """
    validation = validate_squad(squad, squad_targets)

    if len(squad) < SQUAD_RULES["min_size"]:
        return False
    if not validation.can_add_more:
        return True
    if budget_remaining < min_player_price:
        return True

    all_at_ideal = all(
        rs.count >= rs.ideal
        for rs in validation.role_status.values()
    )
    return all_at_ideal


def compute_final_score(squad: List[dict], squad_targets: Dict, playing_xi_shape: Dict) -> Dict:
    """
    Compute end-of-auction squad quality score (0-100 range).
    Mirrors the JS computeFinalScores logic.
    """
    if not squad:
        return {"final_score": 0.0, "grade": "D", "completeness": 0, "star_power": 0.0,
                "balance": 0, "overseas_util": 0}

    role_counts = get_role_counts(squad)

    # Completeness
    roles_with_min = [(r, l) for r, l in squad_targets.items() if l["min"] > 0]
    met = sum(1 for r, l in roles_with_min if role_counts.get(r, 0) >= l["min"])
    completeness = round((met / len(roles_with_min)) * 100) if roles_with_min else 0

    # Playing XI selection → star power (75% XI, 25% bench)
    xi      = _select_playing_xi(squad, playing_xi_shape)
    xi_ids  = {p["id"] for p in xi}
    bench   = [p for p in squad if p["id"] not in xi_ids]
    xi_avg  = sum(p["overall"] for p in xi) / len(xi) if xi else 0
    bench_avg = sum(p["overall"] for p in bench) / len(bench) if bench else xi_avg
    star_power = round(xi_avg * 0.75 + bench_avg * 0.25, 1)

    # Balance (proportional)
    total_ideal = sum(l["ideal"] for l in squad_targets.values())
    balance_scores = []
    for role, limits in squad_targets.items():
        count       = role_counts.get(role, 0)
        actual_share = count / len(squad)
        ideal_share  = limits["ideal"] / total_ideal
        diff         = abs(actual_share - ideal_share)
        balance_scores.append(max(0.0, 1.0 - diff * 12))
    balance = round(sum(balance_scores) / len(balance_scores) * 100)

    # Overseas util
    overseas = sum(1 for p in squad if p.get("nationality") == "Overseas")
    ideal_ratio = SQUAD_RULES["max_overseas"] / 17.0
    actual_ratio = overseas / len(squad)
    overseas_util = round(min(1.0, actual_ratio / ideal_ratio) * 100)

    # Composite
    final_score = round(
        completeness * 0.40
        + max(0, star_power - 70) * 1.75
        + balance * 0.20
        + overseas_util * 0.05
    )

    grade = (
        "S" if final_score >= 80 else
        "A" if final_score >= 65 else
        "B" if final_score >= 50 else
        "C" if final_score >= 35 else "D"
    )

    return {
        "final_score":   final_score,
        "grade":         grade,
        "completeness":  completeness,
        "star_power":    star_power,
        "xi_avg":        round(xi_avg, 1),
        "bench_avg":     round(bench_avg, 1),
        "balance":       balance,
        "overseas_util": overseas_util,
        "squad_size":    len(squad),
        "overseas_count": overseas,
    }


def _select_playing_xi(squad: List[dict], xi_shape: Dict[str, int]) -> List[dict]:
    """Pick best XI using team's tactical shape."""
    by_role = {}
    for p in squad:
        by_role.setdefault(p["role"], []).append(p)
    for r in by_role:
        by_role[r].sort(key=lambda x: -x["overall"])

    selected = []
    used_ids = set()

    for role, slots in xi_shape.items():
        candidates = [p for p in by_role.get(role, []) if p["id"] not in used_ids]
        for p in candidates[:slots]:
            selected.append(p)
            used_ids.add(p["id"])

    # Fill remaining from best leftover
    if len(selected) < 11:
        rest = sorted(
            [p for p in squad if p["id"] not in used_ids],
            key=lambda x: -x["overall"]
        )
        for p in rest:
            if len(selected) >= 11:
                break
            selected.append(p)

    return selected
