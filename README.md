# IPL Auction RL

Multi-agent reinforcement learning environment for the IPL cricket auction.
10 franchise agents learn to bid on 236 real players (built from IPL 2023–2025
and T20 World Cup data) using MAPPO (Multi-Agent PPO).

## Project Overview

This project trains RL agents to play the IPL mega-auction — a sequential
resource allocation game where 10 teams bid on players across 14 auction pools,
managing a ₹100 Crore budget to build the best 15–19 player squad.

**Why this is an interesting RL problem:**
- Sequential decisions under uncertainty (you don't know what rivals will bid)
- Scarcity dynamics (only 17 WK-BATs available for 10 teams)
- Budget management over a long horizon (~200 steps per episode)
- Mixed cooperative/competitive structure (teams aren't trying to hurt each other)
- Rich observation space including squad state, pool scarcity, opponent budgets

## Repository Structure

```
ipl-auction-rl/
├── environment/
│   ├── auction_env.py       # PettingZoo AEC environment
│   ├── pool_generator.py    # Auction pool generation
│   ├── squad_validator.py   # IPL squad rules enforcement
│   └── team_config.py       # 10 team personalities + squad targets
├── agents/
│   ├── mappo_agent.py       # MAPPO actor-critic (Stage 3)
│   └── rule_based_agent.py  # Handcoded baseline (Stage 2)
├── training/
│   ├── trainer.py           # Main training loop (Stage 3)
│   └── rollout_buffer.py    # Experience collection
├── evaluation/
│   ├── evaluator.py         # Evaluation harness
│   └── visualizer.py        # Training curves + squad analysis
├── data/
│   └── players.json         # 236 real players (IPL 2023-25 stats)
├── configs/
│   ├── default.yaml         # Default hyperparameters
│   ├── fast_debug.yaml      # Quick local test
│   └── gpu_full.yaml        # NYU Greene HPC config
└── scripts/
    ├── test_env.py          # Environment smoke test ← start here
    ├── train.py             # Training entry point (Stage 3)
    └── submit_hpc.sh        # NYU Greene SLURM script
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Test the environment (Stage 1)
python scripts/test_env.py

# Run rule-based baseline (Stage 2)
python scripts/run_baseline.py

# Train MAPPO (Stage 3+)
python scripts/train.py --config configs/default.yaml
```

## Training Stages

| Stage | Description | Commits |
|-------|-------------|---------|
| 1 | Environment scaffold + smoke test | `feat: env`, `test: random agents` |
| 2 | Rule-based baseline agent | `feat: rule-based`, `eval: baseline` |
| 3 | MAPPO implementation | `feat: mappo`, `feat: ppo-update` |
| 4 | First training run (CPU, 50k ep) | `train: run-001` |
| 5 | Reward shaping | `feat: reward-shaping`, `train: run-002` |
| 6 | GPU training @ NYU Greene (1M ep) | `train: run-003-gpu` |
| 7 | Ablations: IPPO vs MAPPO, param sharing | `experiment: ablations` |
| 8 | Final evaluation + paper writeup | `eval: final`, `docs: results` |

## Environment Details

**Observation space** (dim=65 per agent):
- Current player features (overall, role, nationality, price, stars)
- Current bid context (bid level, whether leading)
- Own squad state (role counts, budget, overseas slots)
- Role need scores (urgency per role based on squad gaps)
- Pool scarcity (remaining players per role)
- Opponent states (budgets, squad sizes, overseas counts)
- Auction progress + player archetype/trait fit

**Action space**: `Discrete(2)` — PASS (0) or BID (1)

**Reward**: Sparse at episode end based on squad quality score:
- Squad completeness (met all role minimums): 40%
- Playing XI star power (75% XI average, 25% bench): 35%  
- Role balance: 20%
- Overseas utilisation: 5%
- Plus intermediate rewards for meeting role minimums mid-auction

## Player Data

236 real players with stats computed from:
- IPL 2023, 2024, 2025 (219 matches, ball-by-ball)
- T20 World Cup 2022/23, 2024, 2025/26 (132 matches)

Stats are real: Bumrah's 6.33 economy, Kohli's 52.8 batting average,
Head's 172.8 strike rate — all from actual deliveries.

## NYU Greene HPC

```bash
# Submit training job
sbatch scripts/submit_hpc.sh

# Monitor
squeue -u $USER
tail -f results/logs/run_003.log
```

Request: 1× A100 GPU, 8 CPU cores, 32GB RAM, 12-24hr walltime.

## Algorithm: MAPPO

Centralized Training, Decentralized Execution (CTDE):
- **Actor**: MLP(65 → 128 → 128 → 2), one per agent (or shared with parameter sharing)
- **Critic**: MLP(650 → 256 → 128 → 1), sees global state (all agents' obs)
- **Update**: PPO clipped objective + GAE advantage + value normalization

## Citation

If you use this environment in research:
```
@misc{ipl-auction-rl-2025,
  title={IPL Auction as a Multi-Agent RL Environment},
  author={Mehul Goel},
  year={2025},
  url={https://github.com/goel-mehul/ipl-auction-rl}
}
```
