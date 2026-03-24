"""
evaluation/visualizer.py

Plots training curves and evaluation metrics.
Saves to results/plots/.
"""

import json
import numpy as np
from pathlib import Path

PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


TEAM_COLORS = {
    "MI":   "#004B8D", "CSK":  "#F7B700", "RCB":  "#EC1C24",
    "KKR":  "#3A225D", "DC":   "#0078BC", "RR":   "#EA1A85",
    "SRH":  "#F7A721", "PBKS": "#ED1B24", "GT":   "#555555",
    "LSG":  "#A72B2A",
}


def plot_baseline_scores(summary: dict, save_path=None, title="Rule-Based Baseline"):
    if not HAS_MPL:
        print("  matplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    teams  = sorted(summary.keys(), key=lambda t: -summary[t]["final_score_mean"])
    scores = [summary[t]["final_score_mean"] for t in teams]
    stds   = [summary[t]["final_score_std"]  for t in teams]
    colors = [TEAM_COLORS.get(t, "#888") for t in teams]

    # 1. Score bar chart
    ax = axes[0]
    bars = ax.barh(teams, scores, xerr=stds, color=colors, alpha=0.85,
                   error_kw={"linewidth": 1.5, "capsize": 3})
    ax.set_xlabel("Final Score (mean ± std)")
    ax.set_title("Squad Quality Score")
    ax.set_xlim(0, 100)
    for bar, score in zip(bars, scores):
        ax.text(score + 1, bar.get_y() + bar.get_height()/2,
                f"{score:.1f}", va="center", fontsize=8)

    # 2. Budget utilisation
    ax = axes[1]
    spent = [summary[t]["budget_spent_mean"] for t in teams]
    left  = [summary[t]["budget_remain_mean"] for t in teams]
    y     = range(len(teams))
    ax.barh(y, spent, color=colors, alpha=0.85, label="Spent")
    ax.barh(y, left,  left=spent, color="#ddd", alpha=0.6, label="Remaining")
    ax.set_yticks(list(y))
    ax.set_yticklabels(teams)
    ax.set_xlabel("₹ Crore")
    ax.set_title("Budget Utilisation")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8)

    # 3. Role minimums met rate
    ax = axes[2]
    mins_rates = [summary[t]["mins_met_rate"] * 100 for t in teams]
    bars = ax.barh(teams, mins_rates, color=colors, alpha=0.85)
    ax.set_xlabel("% Episodes")
    ax.set_title("Role Minimums Met Rate")
    ax.set_xlim(0, 100)
    ax.axvline(x=100, color="green", linestyle="--", linewidth=1, alpha=0.5)
    for bar, rate in zip(bars, mins_rates):
        ax.text(rate + 1, bar.get_y() + bar.get_height()/2,
                f"{rate:.0f}%", va="center", fontsize=8)

    plt.tight_layout()
    path = save_path or PLOTS_DIR / "baseline_scores.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_score_distribution(per_team: dict, save_path=None, title="Score Distribution"):
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    teams = sorted(per_team.keys(), key=lambda t: -np.mean(per_team[t]["final_score"]))

    for i, team in enumerate(teams):
        scores = per_team[team]["final_score"]
        parts  = ax.violinplot(scores, positions=[i], showmedians=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(TEAM_COLORS.get(team, "#888"))
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(teams)))
    ax.set_xticklabels(teams)
    ax.set_ylabel("Final Score")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = save_path or PLOTS_DIR / "score_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_training_curves(log_path: str, save_path=None):
    """Plot training curves from a JSONL log file."""
    if not HAS_MPL:
        return

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("  No training records found")
        return

    episodes = [r["episode"] for r in records]
    avg_scores = [r.get("avg_score", 0) for r in records]
    avg_rewards = [r.get("avg_reward", 0) for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MAPPO Training Curves", fontsize=13, fontweight="bold")

    # Smooth with rolling window
    def smooth(vals, w=20):
        if len(vals) < w: return vals
        return np.convolve(vals, np.ones(w)/w, mode="valid")

    ax1.plot(episodes, avg_scores, alpha=0.3, color="steelblue", linewidth=0.8)
    if len(avg_scores) >= 20:
        sm_ep = episodes[19:]
        ax1.plot(sm_ep, smooth(avg_scores), color="steelblue", linewidth=2, label="Smoothed")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean Squad Score")
    ax1.set_title("Squad Quality Over Training")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(episodes, avg_rewards, alpha=0.3, color="coral", linewidth=0.8)
    if len(avg_rewards) >= 20:
        ax2.plot(episodes[19:], smooth(avg_rewards), color="coral", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Episode Reward Over Training")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = save_path or PLOTS_DIR / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
