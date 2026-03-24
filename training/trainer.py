"""
training/trainer.py

MAPPO training loop.

Flow per iteration:
  1. Collect rollout_steps across num_envs parallel episodes
  2. Compute GAE advantages
  3. Run ppo_epochs × minibatch updates
  4. Log metrics, save checkpoints

Key hyperparameters (from configs/default.yaml):
  gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2
  ppo_epochs=4, minibatch_size=64, lr=3e-4
"""

import os
import sys
import json
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.auction_env import IPLAuctionEnv
from environment.team_config import ALL_TEAM_IDS
from agents.mappo_agent import MAPPOAgent
from training.rollout_buffer import RolloutBuffer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class MAPPOTrainer:
    """
    Trains a shared MAPPO policy across all 10 IPL franchise agents.

    Parameter sharing: one Actor + Critic for all teams.
    Centralised training: Critic sees full 650-dim global state.
    Decentralised execution: Actor uses only own 65-dim obs.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self.cfg    = config
        self.device = device
        tc = config["training"]
        nc = config["network"]
        rc = config["reward"]

        # ── Environment ───────────────────────────────────────────
        self.env = IPLAuctionEnv(
            retentions_enabled=True,
            reward_config={
                "final_score_weight":  rc["final_score_weight"],
                "completeness_bonus":  rc["completeness_bonus"],
                "budget_waste_penalty":rc["budget_waste_penalty"],
                "overseas_bonus":      rc["overseas_bonus"],
            }
        )

        # ── Agent ─────────────────────────────────────────────────
        self.agent = MAPPOAgent(
            obs_dim=nc["obs_dim"],
            global_state_dim=nc["global_state_dim"],
            hidden_dim=nc["hidden_dim"],
            num_layers=nc["num_layers"],
            activation=nc["activation"],
            use_orthogonal=nc["use_orthogonal_init"],
            value_normalize=tc["value_normalize"],
            device=device,
        )
        print(f"  Agent parameters: {self.agent.count_parameters():,}")

        # ── Optimiser ─────────────────────────────────────────────
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=tc["lr"],
            eps=1e-5,
        )
        if tc.get("lr_decay"):
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=tc["total_episodes"],
            )
        else:
            self.scheduler = None

        # ── Buffer ────────────────────────────────────────────────
        self.buffer = RolloutBuffer(
            num_agents=len(ALL_TEAM_IDS),
            gamma=tc["gamma"],
            gae_lambda=tc["gae_lambda"],
        )
        for team_id in ALL_TEAM_IDS:
            self.buffer.init_agent(team_id)

        # ── Logging ───────────────────────────────────────────────
        log_cfg  = config["logging"]
        self.results_dir   = Path(log_cfg["results_dir"])
        self.ckpt_dir      = self.results_dir / "checkpoints"
        self.logs_dir      = self.results_dir / "logs"
        self.plots_dir     = self.results_dir / "plots"
        for d in [self.ckpt_dir, self.logs_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.log_path = self.logs_dir / "training_log.jsonl"
        self.log_file = open(self.log_path, "a")

        # ── Training state ────────────────────────────────────────
        self.episode      = 0
        self.total_steps  = 0
        self.best_score   = -float("inf")
        self.score_history= []

        # ── Parallel env setup ────────────────────────────────────
        self.num_envs = tc.get("num_envs", 1)
        if self.num_envs > 1:
            from training.vec_env import VecEnv
            self.vec_env = VecEnv(num_envs=self.num_envs, seed=0)
            print(f"  Parallel envs:   {self.num_envs}")
        else:
            self.vec_env = None

        # Hyperparams shortcut
        self.tc = tc
        self.lc = log_cfg

    # ── Rollout collection ─────────────────────────────────────────

    def _make_action_fn(self):
        """Creates a picklable action function for parallel workers."""
        # We can't pickle the MAPPOAgent directly (has MPS/CUDA tensors)
        # So we move weights to CPU, extract as numpy, and rebuild in worker
        # For simplicity: use the agent on main process for inference,
        # and only parallelize the environment stepping
        agent = self.agent
        agent.eval()

        def action_fn(obs, global_state):
            action, log_prob = agent.get_action(obs)
            value            = agent.get_value(global_state)
            return action, log_prob, value

        return action_fn

    def collect_rollout(self, n_episodes: int = 1) -> dict:
        """
        Run n_episodes, store transitions in buffer.
        Uses parallel envs if num_envs > 1.
        """
        self.buffer.reset_all()
        self.agent.eval()

        ep_scores  = []
        ep_steps   = []
        ep_rewards = defaultdict(list)

        if self.num_envs > 1 and self.vec_env is not None:
            return self._collect_parallel(n_episodes, ep_scores, ep_steps, ep_rewards)

        return self._collect_sequential(n_episodes, ep_scores, ep_steps, ep_rewards)

    def _collect_parallel(self, n_episodes, ep_scores, ep_steps, ep_rewards):
        """Parallel rollout using multiprocessing."""
        from training.vec_env import _run_single_episode
        import multiprocessing as mp
        import functools

        # Move agent to CPU for pickling to workers
        self.agent.to("cpu")
        action_fn = self._make_action_fn()

        seeds = list(range(self.episode, self.episode + n_episodes))
        n_workers = min(self.num_envs, n_episodes, mp.cpu_count())

        with mp.Pool(processes=n_workers) as pool:
            worker_fn = functools.partial(_run_single_episode, get_action_fn=action_fn)
            results   = pool.map(worker_fn, seeds)

        # Move agent back to original device
        self.agent.to(self.device)
        self.agent.train()

        # Load results into buffer
        for result in results:
            for team_id in ALL_TEAM_IDS:
                for t in result["step_data"][team_id]:
                    self.buffer.add(
                        team_id,
                        t["obs"], t["global_state"],
                        t["action"], t["reward"],
                        t["done"], t["log_prob"], t["value"],
                    )
            if result["scores"]:
                ep_scores.append(result["mean_score"])
            ep_steps.append(result["steps"])
            self.episode    += 1
            self.total_steps += result["steps"]

        return {
            "mean_score":  float(np.mean(ep_scores)) if ep_scores else 0.0,
            "mean_reward": 0.0,
            "mean_steps":  float(np.mean(ep_steps)),
            "buffer_size": self.buffer.total_steps(),
        }

    def _collect_sequential(self, n_episodes, ep_scores, ep_steps, ep_rewards):
        """Sequential rollout — one episode at a time."""
        for ep in range(n_episodes):
            obs_dict, _ = self.env.reset(seed=self.episode + ep)
            step = 0
            step_data = {t: [] for t in ALL_TEAM_IDS}

            while self.env.agents and step < 30000:
                agent_id = self.env.agent_selection
                obs, reward, term, trunc, info = self.env.last()

                if term or trunc:
                    if step_data[agent_id]:
                        step_data[agent_id][-1]["done"]   = True
                        step_data[agent_id][-1]["reward"] += reward
                    self.env.step(None)
                    step += 1
                    continue

                global_state = self.env.state()
                action, log_prob = self.agent.get_action(obs)
                value            = self.agent.get_value(global_state)

                step_data[agent_id].append({
                    "obs":          obs.copy(),
                    "global_state": global_state.copy(),
                    "action":       action,
                    "log_prob":     log_prob,
                    "value":        value,
                    "reward":       reward,
                    "done":         False,
                })

                self.env.step(action)
                step += 1

            scores = self.env.get_final_scores()

            for team_id in ALL_TEAM_IDS:
                transitions = step_data[team_id]
                if not transitions:
                    continue

                final_reward = self.env._cumulative_rewards.get(team_id, 0.0)
                ep_rewards[team_id].append(final_reward)
                transitions[-1]["reward"] += final_reward

                for t in transitions:
                    self.buffer.add(
                        team_id,
                        t["obs"], t["global_state"],
                        t["action"], t["reward"],
                        t["done"], t["log_prob"], t["value"],
                    )

            if scores:
                ep_score = np.mean([s.get("final_score", 0) for s in scores.values()])
                ep_scores.append(ep_score)
            ep_steps.append(step)
            self.episode    += 1
            self.total_steps += step

        self.agent.train()
        return {
            "mean_score":  float(np.mean(ep_scores)) if ep_scores else 0.0,
            "mean_reward": float(np.mean([v for vals in ep_rewards.values() for v in vals])),
            "mean_steps":  float(np.mean(ep_steps)),
            "buffer_size": self.buffer.total_steps(),
        }


    # ── PPO update ─────────────────────────────────────────────────

    def update(self) -> dict:
        """
        Run ppo_epochs passes over the collected data.
        Returns loss metrics.
        """
        data = self.buffer.get_all_data(ALL_TEAM_IDS)
        if data is None:
            return {}

        # Convert to tensors
        obs_t   = torch.FloatTensor(data["obs"]).to(self.device)
        gs_t    = torch.FloatTensor(data["global_state"]).to(self.device)
        act_t   = torch.LongTensor(data["actions"]).to(self.device)
        adv_t   = torch.FloatTensor(data["advantages"]).to(self.device)
        ret_t   = torch.FloatTensor(data["returns"]).to(self.device)
        old_lp  = torch.FloatTensor(data["old_log_probs"]).to(self.device)

        # Normalise value targets
        if self.agent.value_normalize:
            ret_np = ret_t.cpu().numpy()
            ret_t  = torch.FloatTensor(
                self.agent.normalize_values(ret_np)
            ).to(self.device)

        tc = self.tc
        all_pg_loss = []
        all_v_loss  = []
        all_entropy = []
        all_approx_kl = []

        for epoch in range(tc["ppo_epochs"]):
            for batch in self.buffer.make_minibatches(data, tc["minibatch_size"]):
                b_obs  = torch.FloatTensor(batch["obs"]).to(self.device)
                b_gs   = torch.FloatTensor(batch["global_state"]).to(self.device)
                b_act  = torch.LongTensor(batch["actions"]).to(self.device)
                b_adv  = torch.FloatTensor(batch["advantages"]).to(self.device)
                b_ret  = torch.FloatTensor(batch["returns"]).to(self.device)
                b_olp  = torch.FloatTensor(batch["old_log_probs"]).to(self.device)

                # Normalise returns for this batch
                if self.agent.value_normalize:
                    b_ret = torch.FloatTensor(
                        self.agent.normalize_values(b_ret.cpu().numpy())
                    ).to(self.device)

                # Forward pass
                log_probs, values, entropy = self.agent.evaluate_for_update(
                    b_obs, b_gs, b_act
                )

                # ── Policy loss (clipped PPO) ──────────────────────
                ratio      = torch.exp(log_probs - b_olp)
                surr1      = ratio * b_adv
                surr2      = torch.clamp(ratio, 1 - tc["clip_epsilon"],
                                                1 + tc["clip_epsilon"]) * b_adv
                pg_loss    = -torch.min(surr1, surr2).mean()

                # ── Value loss (clipped) ───────────────────────────
                v_loss     = F.mse_loss(values, b_ret)

                # ── Entropy bonus ──────────────────────────────────
                ent_loss   = -entropy.mean()

                # ── Total loss ─────────────────────────────────────
                loss = (pg_loss
                        + tc["value_coef"]   * v_loss
                        + tc["entropy_coef"] * ent_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), tc["max_grad_norm"]
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    all_approx_kl.append(approx_kl)

                all_pg_loss.append(pg_loss.item())
                all_v_loss.append(v_loss.item())
                all_entropy.append(entropy.mean().item())

        if self.scheduler:
            self.scheduler.step()

        return {
            "pg_loss":    float(np.mean(all_pg_loss)),
            "value_loss": float(np.mean(all_v_loss)),
            "entropy":    float(np.mean(all_entropy)),
            "approx_kl":  float(np.mean(all_approx_kl)),
            "lr":         self.optimizer.param_groups[0]["lr"],
        }

    # ── Main training loop ─────────────────────────────────────────

    def train(self):
        tc  = self.tc
        lc  = self.lc
        eps_per_iter = tc.get("num_envs", 1)   # episodes per collect call
        total_iters  = tc["total_episodes"] // eps_per_iter

        print(f"\n  Starting MAPPO training")
        print(f"  Total episodes:  {tc['total_episodes']}")
        print(f"  Episodes/iter:   {eps_per_iter}")
        print(f"  Total iters:     {total_iters}")
        print(f"  Device:          {self.device}")
        print()

        t_start = time.time()

        for iteration in range(total_iters):
            # 1. Collect rollout
            rollout_stats = self.collect_rollout(n_episodes=eps_per_iter)

            # 2. PPO update
            update_stats  = self.update()

            # 3. Track score
            score = rollout_stats["mean_score"]
            self.score_history.append(score)

            # 4. Log
            log_entry = {
                "episode":    self.episode,
                "iteration":  iteration,
                "avg_score":  score,
                "avg_reward": rollout_stats["mean_reward"],
                "buffer_size":rollout_stats["buffer_size"],
                **update_stats,
            }
            self.log_file.write(json.dumps(log_entry) + "\n")
            self.log_file.flush()

            # 5. Console log
            if (iteration + 1) % lc["log_interval"] == 0:
                elapsed = time.time() - t_start
                ep_per_s = self.episode / elapsed
                print(
                    f"  iter={iteration+1:5d}  "
                    f"ep={self.episode:6d}  "
                    f"score={score:5.1f}  "
                    f"reward={rollout_stats['mean_reward']:6.3f}  "
                    f"pg={update_stats.get('pg_loss',0):6.3f}  "
                    f"vf={update_stats.get('value_loss',0):6.3f}  "
                    f"ent={update_stats.get('entropy',0):5.3f}  "
                    f"kl={update_stats.get('approx_kl',0):.4f}  "
                    f"ep/s={ep_per_s:.2f}"
                )

            # 6. Save checkpoint
            if (iteration + 1) % lc["save_interval"] == 0 or iteration == total_iters - 1:
                ckpt_path = self.ckpt_dir / f"checkpoint_ep{self.episode}.pt"
                self.agent.save(str(ckpt_path))

                # Save best
                if score > self.best_score:
                    self.best_score = score
                    self.agent.save(str(self.ckpt_dir / "best.pt"))
                    print(f"  ★ New best score: {score:.2f} → saved best.pt")

        # Final save
        self.agent.save(str(self.ckpt_dir / "final.pt"))
        self.log_file.close()

        elapsed = time.time() - t_start
        print(f"\n  Training complete in {elapsed/3600:.2f}h")
        print(f"  Best score: {self.best_score:.2f}")
        print(f"  Final checkpoint: {self.ckpt_dir}/final.pt")

        return {
            "best_score":    self.best_score,
            "total_episodes": self.episode,
            "elapsed_s":     elapsed,
        }
