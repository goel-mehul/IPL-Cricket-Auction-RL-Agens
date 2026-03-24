"""
scripts/run_baseline.py

Evaluates the rule-based agent baseline.

Run:
    python scripts/run_baseline.py --episodes 100    # quick test
    python scripts/run_baseline.py --episodes 1000   # full baseline (~30 min)
"""

import sys, json, time, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.auction_env import IPLAuctionEnv
from environment.pool_generator import get_remaining_role_counts
from environment.team_config import ALL_TEAM_IDS
from agents.rule_based_agent import RuleBasedAgent
from evaluation.evaluator import Evaluator, print_summary
from evaluation.visualizer import plot_baseline_scores, plot_score_distribution


def make_agent_fn(agents_dict):
    def fn(team_id, obs, env):
        agent  = agents_dict[team_id]
        s      = env.get_state_dict()
        player = env._current_player(s["pools"], s["pool_idx"], s["player_idx"])
        if player is None:
            return 0
        team  = s["teams"][team_id]
        return agent.act(
            player=player,
            squad=team["squad"],
            budget=team["budget"],
            remaining_role_counts=get_remaining_role_counts(
                s["pools"], s["pool_idx"], s["player_idx"]
            ),
            current_bid=s["current_bid_cr"],
            leading_team=s["leading_team"],
        )
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int, default=100)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--no-plots",   action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    print("=" * 65)
    print("  IPL Auction RL — Rule-Based Baseline Evaluation")
    print(f"  Episodes: {args.episodes}  Seed: {args.seed}")
    print("=" * 65)

    env   = IPLAuctionEnv(retentions_enabled=True)
    master_rng = np.random.default_rng(args.seed)
    agents = {
        t: RuleBasedAgent(t, rng=np.random.default_rng(master_rng.integers(int(1e9))))
        for t in ALL_TEAM_IDS
    }

    evaluator = Evaluator(env)
    t0 = time.time()
    results = evaluator.evaluate(
        agent_fn=make_agent_fn(agents),
        n_episodes=args.episodes,
        seed_start=args.seed,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n  Done in {elapsed:.1f}s  ({elapsed/args.episodes:.2f}s/ep)  "
          f"timeouts={results['n_timeouts']}")

    print_summary(results["summary"],
                  title=f"Rule-Based Baseline ({args.episodes} episodes)")

    all_scores = [s for t in ALL_TEAM_IDS
                    for s in results["per_team"][t]["final_score"]]
    print(f"  All-team score: mean={np.mean(all_scores):.2f}  "
          f"std={np.std(all_scores):.2f}  "
          f"min={np.min(all_scores):.0f}  max={np.max(all_scores):.0f}\n")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {
        "agent":              "rule_based",
        "n_episodes":         results["n_episodes"],
        "n_timeouts":         results["n_timeouts"],
        "mean_ep_time_s":     results["mean_ep_time"],
        "overall_score_mean": float(np.mean(all_scores)),
        "overall_score_std":  float(np.std(all_scores)),
        "summary":            results["summary"],
    }
    path = out_dir / "baseline_results.json"
    with open(path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"  Saved → {path}")

    if not args.no_plots:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        print("  Generating plots...")
        plot_baseline_scores(results["summary"],
                             save_path=plots_dir / "baseline_scores.png")
        plot_score_distribution(results["per_team"],
                                save_path=plots_dir / "score_distribution.png")

    print("\n  Commit commands:")
    print("    git add results/")
    print(f"    git commit -m 'eval: rule-based baseline {args.episodes} episodes'")
    print("    git push")


if __name__ == "__main__":
    main()
