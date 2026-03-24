"""
evaluation/evaluator.py

Reusable evaluation harness.
Runs N episodes with a given set of agents and collects metrics.
"""

import time
import numpy as np
from typing import Dict, List, Callable, Optional
from collections import defaultdict

from environment.auction_env import IPLAuctionEnv
from environment.team_config import ALL_TEAM_IDS, TEAM_CONFIGS, RETENTION_SLABS


def run_episode_with_agents(env, agent_fn, seed=None, max_steps=30000):
    """
    Run one full episode.
    agent_fn: dict mapping team_id → callable(agent_id, obs, env) → action
    """
    obs_dict, _ = env.reset(seed=seed)
    steps = 0

    while env.agents and steps < max_steps:
        agent = env.agent_selection
        obs, reward, term, trunc, info = env.last()

        if term or trunc:
            env.step(None)
        else:
            fn = agent_fn.get(agent) if isinstance(agent_fn, dict) else agent_fn
            action = fn(agent, obs, env) if fn else 0
            env.step(action)

        steps += 1

    return {
        "scores":  env.get_final_scores(),
        "state":   env.get_state_dict(),
        "steps":   steps,
        "timeout": steps >= max_steps,
    }


class Evaluator:
    def __init__(self, env):
        self.env = env

    def evaluate(self, agent_fn, n_episodes=100, seed_start=0, verbose=True):
        per_team      = {t: defaultdict(list) for t in ALL_TEAM_IDS}
        episode_times = []
        n_timeouts    = 0

        for ep in range(n_episodes):
            t0      = time.time()
            result  = run_episode_with_agents(self.env, agent_fn, seed=seed_start + ep)
            elapsed = time.time() - t0
            episode_times.append(elapsed)
            if result["timeout"]:
                n_timeouts += 1

            state = result["state"]
            for team_id in ALL_TEAM_IDS:
                sc    = result["scores"].get(team_id, {})
                team  = state["teams"][team_id]
                squad = team["squad"]

                per_team[team_id]["final_score"].append(sc.get("final_score", 0))
                per_team[team_id]["grade"].append(sc.get("grade", "D"))
                per_team[team_id]["squad_size"].append(sc.get("squad_size", 0))
                per_team[team_id]["xi_avg"].append(sc.get("xi_avg", 0))
                per_team[team_id]["completeness"].append(sc.get("completeness", 0))
                per_team[team_id]["balance"].append(sc.get("balance", 0))
                per_team[team_id]["overseas_count"].append(sc.get("overseas_count", 0))
                per_team[team_id]["budget_remaining"].append(team["budget"])
                num_ret  = sum(1 for p in squad if p.get("is_retained", False))
                ret_cost = sum(RETENTION_SLABS[:num_ret])
                per_team[team_id]["budget_spent"].append(100.0 - team["budget"] - ret_cost)

                from environment.squad_validator import validate_squad
                v = validate_squad(squad, TEAM_CONFIGS[team_id]["squad_targets"])
                per_team[team_id]["mins_met"].append(1.0 if v.all_mins_met else 0.0)

            if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
                avg = np.mean([per_team[t]["final_score"][-1] for t in ALL_TEAM_IDS])
                print(f"  Episode {ep+1:4d}/{n_episodes}  avg_score={avg:.1f}  time={elapsed:.2f}s")

        # Build summary
        summary = {}
        for team_id in ALL_TEAM_IDS:
            d = per_team[team_id]
            grades = d["grade"]
            summary[team_id] = {
                "final_score_mean":  float(np.mean(d["final_score"])),
                "final_score_std":   float(np.std(d["final_score"])),
                "xi_avg_mean":       float(np.mean(d["xi_avg"])),
                "squad_size_mean":   float(np.mean(d["squad_size"])),
                "completeness_mean": float(np.mean(d["completeness"])),
                "balance_mean":      float(np.mean(d["balance"])),
                "budget_spent_mean": float(np.mean(d["budget_spent"])),
                "budget_remain_mean":float(np.mean(d["budget_remaining"])),
                "mins_met_rate":     float(np.mean(d["mins_met"])),
                "overseas_mean":     float(np.mean(d["overseas_count"])),
                "grade_dist":        {g: round(grades.count(g)/len(grades), 3)
                                      for g in ["S","A","B","C","D"]},
            }

        return {
            "per_team":       per_team,
            "summary":        summary,
            "episode_times":  episode_times,
            "n_episodes":     n_episodes,
            "n_timeouts":     n_timeouts,
            "mean_ep_time":   float(np.mean(episode_times)),
        }


def print_summary(summary, title="Evaluation Summary"):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")
    print(f"  {'TEAM':<8} {'SCORE':>7} {'±':>5} {'XI_AVG':>7} {'SQD':>5} "
          f"{'MINS%':>6} {'SPENT':>7}  GRADES")
    print(f"  {'-'*68}")
    for team_id, s in sorted(summary.items(), key=lambda x: -x[1]["final_score_mean"]):
        gd = s["grade_dist"]
        gs = " ".join(f"{g}:{gd[g]:.0%}" for g in ["S","A","B","C","D"] if gd.get(g,0)>0)
        print(f"  {team_id:<8} "
              f"{s['final_score_mean']:>7.1f} "
              f"{s['final_score_std']:>5.1f} "
              f"{s['xi_avg_mean']:>7.1f} "
              f"{s['squad_size_mean']:>5.1f} "
              f"{s['mins_met_rate']:>6.0%} "
              f"{s['budget_spent_mean']:>7.1f}  "
              f"{gs}")
    print(f"{'='*68}\n")
