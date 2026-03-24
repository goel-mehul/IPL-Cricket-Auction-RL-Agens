"""
scripts/train.py

MAPPO training entry point.

Usage:
    python scripts/train.py                              # default config
    python scripts/train.py --config configs/fast_debug.yaml   # quick test
    python scripts/train.py --config configs/gpu_full.yaml     # NYU HPC
    python scripts/train.py --resume results/checkpoints/checkpoint_ep500.pt
"""

import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import MAPPOTrainer, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu / cuda / mps (auto-detected if not set)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override total_episodes from config")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.episodes:
        config["training"]["total_episodes"] = args.episodes

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
        print("  Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("  Using Apple MPS")
    else:
        device = "cpu"
        print("  Using CPU")

    print(f"\n{'='*60}")
    print(f"  IPL Auction RL — MAPPO Training")
    print(f"  Config:   {args.config}")
    print(f"  Device:   {device}")
    print(f"  Episodes: {config['training']['total_episodes']}")
    print(f"{'='*60}")

    # Build trainer
    trainer = MAPPOTrainer(config=config, device=device)

    # Resume from checkpoint
    if args.resume:
        trainer.agent.load(args.resume)
        print(f"  Resumed from: {args.resume}")

    # Train
    result = trainer.train()

    print(f"\n  Done!")
    print(f"  Best score:  {result['best_score']:.2f}")
    print(f"  Total eps:   {result['total_episodes']}")
    print(f"  Time:        {result['elapsed_s']/3600:.2f}h")
    print()
    eps = config["training"]["total_episodes"]
    print("  Commit commands:")
    print(f"    git add results/")
    print(f"    git commit -m 'train: MAPPO run-001 {eps} episodes'")
    print(f"    git push")


if __name__ == "__main__":
    main()
