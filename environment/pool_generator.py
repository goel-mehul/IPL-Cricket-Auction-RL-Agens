"""
pool_generator.py

Port of poolGenerator.js.
Splits 236 players into ordered auction pools:
  1. Marquee pool   — top 20 by overall
  2. Role pools     — BAT / WK-BAT / ALL / SPIN / PACE sets of up to 20
  
Within each pool, order is shuffled for unpredictability.
"""

import random
from typing import List, Set, Dict, Optional

POOL_SIZE      = 20
ROLE_POOL_ORDER = ["BAT", "WK-BAT", "ALL", "SPIN", "PACE"]

ROLE_LABELS = {
    "BAT":    "Batsmen",
    "WK-BAT": "Wicket-Keepers",
    "ALL":    "All-Rounders",
    "SPIN":   "Spinners",
    "PACE":   "Fast Bowlers",
}


def generate_pools(
    all_players: List[dict],
    retained_ids: Optional[Set[int]] = None,
    seed: Optional[int] = None,
) -> List[dict]:
    """
    Generate ordered auction pools from player list.

    Args:
        all_players:   Full player list
        retained_ids:  Set of player IDs to exclude (already retained)
        seed:          Random seed for reproducibility during training

    Returns:
        List of pool dicts: { id, label, role, set_number, players }
    """
    rng = random.Random(seed)

    retained_ids = retained_ids or set()
    available = [p for p in all_players if p["id"] not in retained_ids]

    # Sort by overall desc for pool assignment
    sorted_players = sorted(available, key=lambda p: -p.get("overall", 0))

    # ── Marquee pool ─────────────────────────────────────────────
    marquee_players = sorted_players[:POOL_SIZE]
    marquee_ids     = {p["id"] for p in marquee_players}
    shuffled_marquee = marquee_players[:]
    rng.shuffle(shuffled_marquee)

    pools = [{
        "id":         "marquee",
        "label":      "Marquee Players",
        "role":       None,
        "set_number": 1,
        "players":    shuffled_marquee,
    }]

    # ── Role pools ────────────────────────────────────────────────
    for role in ROLE_POOL_ORDER:
        role_players = [
            p for p in sorted_players
            if p.get("role") == role and p["id"] not in marquee_ids
        ]

        set_number = 1
        i = 0
        while i < len(role_players):
            chunk = role_players[i: i + POOL_SIZE]
            shuffled = chunk[:]
            rng.shuffle(shuffled)
            pools.append({
                "id":         f"{role.lower().replace('-', '_')}_set_{set_number}",
                "label":      f"{ROLE_LABELS[role]} — Set {set_number}",
                "role":       role,
                "set_number": set_number,
                "players":    shuffled,
            })
            set_number += 1
            i += POOL_SIZE

    return pools


def flatten_pools(pools: List[dict]) -> List[dict]:
    """Returns all players in auction order."""
    return [p for pool in pools for p in pool["players"]]


def get_remaining_role_counts(
    pools: List[dict],
    current_pool_idx: int,
    current_player_idx: int,
) -> Dict[str, int]:
    """
    How many of each role remain in the auction from this point forward?
    Used as scarcity signal in the observation.
    """
    from environment.squad_validator import ROLES
    counts = {r: 0 for r in ROLES}

    for pool_idx, pool in enumerate(pools):
        for player_idx, player in enumerate(pool["players"]):
            if pool_idx < current_pool_idx:
                continue
            if pool_idx == current_pool_idx and player_idx <= current_player_idx:
                continue
            role = player.get("role", "")
            if role in counts:
                counts[role] += 1

    return counts


def get_pool_summary(pools: List[dict]) -> List[dict]:
    return [
        {
            "id":           pool["id"],
            "label":        pool["label"],
            "role":         pool["role"],
            "player_count": len(pool["players"]),
        }
        for pool in pools
    ]
