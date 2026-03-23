"""
test_env.py

Smoke test: run a full auction episode with random agents.
Verifies the environment terminates correctly and all squads are valid.

Run with:
    python scripts/test_env.py
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.auction_env import IPLAuctionEnv, OBS_DIM
from environment.squad_validator import validate_squad, SQUAD_RULES
from environment.team_config import TEAM_CONFIGS, ALL_TEAM_IDS


def run_episode(env: IPLAuctionEnv, verbose: bool = False) -> dict:
    """Run one full auction episode with random agents. Returns final scores."""
    obs_dict, infos = env.reset(seed=42)
    step_count = 0
    max_steps  = 25000  # 206 players × ~30 steps each (bidding) + unsold round

    while env.agents and step_count < max_steps:
        agent = env.agent_selection
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            env.step(None)
        else:
            # Random policy: bid 30% of the time
            action = 1 if np.random.random() < 0.3 else 0
            env.step(action)

        step_count += 1

    if step_count >= max_steps:
        print(f"  WARNING: hit max_steps ({max_steps})")

    scores = env.get_final_scores()
    return scores, step_count


def validate_squads(env: IPLAuctionEnv) -> bool:
    """Check all squads satisfy basic rules."""
    state   = env.get_state_dict()
    all_ok  = True
    for team_id in ALL_TEAM_IDS:
        team   = state["teams"][team_id]
        config = TEAM_CONFIGS[team_id]
        squad  = team["squad"]
        v      = validate_squad(squad, config["squad_targets"])

        size_ok     = SQUAD_RULES["min_size"] <= len(squad) <= SQUAD_RULES["max_size"]
        overseas    = sum(1 for p in squad if p.get("nationality") == "Overseas")
        overseas_ok = overseas <= SQUAD_RULES["max_overseas"]
        budget_ok   = team["budget"] >= 0

        status = "✓" if (size_ok and overseas_ok and budget_ok) else "✗"
        if status == "✗":
            all_ok = False

    print(f"  {status} {team_id:6s}  squad={len(squad):2d}  "
          f"overseas={overseas}  budget=₹{team['budget']:.1f}Cr  "
          f"mins_met={v.all_mins_met}"
          + ("  (bankrupt)" if team['budget'] < 0.5 and len(squad) < SQUAD_RULES["min_size"] else ""))
    return all_ok


def test_observation_shape(env: IPLAuctionEnv):
    """Verify obs dimensions and bounds."""
    obs_dict, _ = env.reset(seed=0)
    for agent, obs in obs_dict.items():
        assert obs.shape == (OBS_DIM,), f"Bad obs shape for {agent}: {obs.shape}"
        assert obs.dtype == np.float32,  f"Bad obs dtype for {agent}: {obs.dtype}"
        assert obs.min() >= 0.0,         f"Obs out of bounds (min) for {agent}"
        assert obs.max() <= 1.0 + 1e-5,  f"Obs out of bounds (max) for {agent}"
    print(f"  ✓ Observation shape: ({OBS_DIM},) float32, all in [0,1]")


def test_state_shape(env: IPLAuctionEnv):
    """Verify global state shape."""
    env.reset(seed=0)
    state = env.state()
    expected = OBS_DIM * len(ALL_TEAM_IDS)
    assert state.shape == (expected,), f"Bad state shape: {state.shape}"
    print(f"  ✓ Global state shape: ({expected},) = 10 × {OBS_DIM}")


def test_determinism(env: IPLAuctionEnv):
    """Same seed → same pool order."""
    np.random.seed(0)
    s1, _ = env.reset(seed=123)
    p1    = env.get_state_dict()["pools"][0]["players"][0]["name"]

    np.random.seed(0)
    s2, _ = env.reset(seed=123)
    p2    = env.get_state_dict()["pools"][0]["players"][0]["name"]

    assert p1 == p2, f"Non-deterministic: {p1} != {p2}"
    print(f"  ✓ Determinism check passed (seed=123 → first player: {p1})")


def test_full_episode(env: IPLAuctionEnv):
    """Full episode with random agents."""
    print("\n  Running full episode (random agents)...")
    t0 = time.time()
    np.random.seed(42)
    scores, steps = run_episode(env, verbose=False)
    elapsed = time.time() - t0

    print(f"  ✓ Episode completed in {steps} steps ({elapsed:.2f}s)")
    print(f"  ✓ {elapsed / steps * 1000:.3f}ms per step")
    print()
    print("  Final squad validation:")
    all_ok = validate_squads(env)

    print()
    print("  Team scores (sorted):")
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1]["final_score"])
    for team_id, sc in sorted_scores:
        print(f"    {team_id:6s}  score={sc['final_score']:3d}  "
              f"grade={sc['grade']}  "
              f"xi_avg={sc['xi_avg']:.1f}  "
              f"squad={sc['squad_size']}")

    return all_ok, scores


def main():
    print("=" * 60)
    print("IPL Auction RL — Environment Smoke Test")
    print("=" * 60)

    env = IPLAuctionEnv(retentions_enabled=True, seed=42)

    print("\n[1] Observation shape test")
    test_observation_shape(env)

    print("\n[2] Global state shape test")
    test_state_shape(env)

    print("\n[3] Determinism test")
    test_determinism(env)

    print("\n[4] Full episode test")
    all_ok, scores = test_full_episode(env)

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ ALL TESTS PASSED — environment is ready for training")
    else:
        print("✗ SOME SQUAD VIOLATIONS — check environment logic")
    print("=" * 60)

    # Speed benchmark
    print("\n[5] Speed benchmark (10 episodes)...")
    t0 = time.time()
    for i in range(10):
        np.random.seed(i)
        run_episode(env)
    elapsed = time.time() - t0
    print(f"  10 episodes in {elapsed:.2f}s → {elapsed/10*1000:.0f}ms/episode")
    print(f"  Estimated training speed: ~{3600/elapsed*10:.0f} episodes/hour")


if __name__ == "__main__":
    main()
