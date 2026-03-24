"""
training/vec_env.py

Vectorized environment — runs N auction environments in parallel
using Python multiprocessing. Each worker process runs its own
IPLAuctionEnv independently.

With 16 workers on 32 cores:
  - Sequential: 4.3s × 2000 episodes = 2.4 hours
  - Parallel:   4.3s / 16 workers × 2000 = 9 minutes effective throughput
  
Usage:
    vec_env = VecEnv(num_envs=16, seed=0)
    results = vec_env.run_episodes(agent_fn, n_episodes=16)
    # Returns 16 episodes worth of experience in ~4.3s
"""

import numpy as np
import multiprocessing as mp
from typing import Callable, List, Dict, Optional
from environment.auction_env import IPLAuctionEnv
from environment.pool_generator import get_remaining_role_counts
from environment.team_config import ALL_TEAM_IDS


def _worker(
    worker_id:    int,
    task_queue:   mp.Queue,
    result_queue: mp.Queue,
):
    """
    Worker process — runs episodes and returns collected transitions.
    Receives (seed, policy_weights) from task_queue.
    Sends episode results back via result_queue.
    """
    env = IPLAuctionEnv(retentions_enabled=True)

    while True:
        task = task_queue.get()
        if task is None:  # poison pill — shut down
            break

        seed, actions_map = task  # actions_map: team_id → list of (obs→action) decisions

        obs_dict, _ = env.reset(seed=seed)
        step_data    = {t: [] for t in ALL_TEAM_IDS}
        steps        = 0

        while env.agents and steps < 30000:
            agent_id = env.agent_selection
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                if step_data[agent_id]:
                    step_data[agent_id][-1]["done"]   = True
                    step_data[agent_id][-1]["reward"] += reward
                env.step(None)
                steps += 1
                continue

            # Use pre-computed actions passed from main process
            # (avoids sending neural network weights to workers)
            action = actions_map.get((agent_id, steps), 0)

            global_state = env.state()
            step_data[agent_id].append({
                "obs":          obs.copy(),
                "global_state": global_state.copy(),
                "action":       action,
                "reward":       reward,
                "done":         False,
                "step":         steps,
            })

            env.step(action)
            steps += 1

        # Collect final rewards
        scores = env.get_final_scores()
        state  = env.get_state_dict()
        for team_id in ALL_TEAM_IDS:
            final_reward = env._cumulative_rewards.get(team_id, 0.0)
            if step_data[team_id]:
                step_data[team_id][-1]["reward"] += final_reward

        result_queue.put({
            "worker_id":  worker_id,
            "seed":       seed,
            "step_data":  step_data,
            "scores":     scores,
            "steps":      steps,
        })


class VecEnv:
    """
    Vectorized environment using multiprocessing.

    For the RL training loop, the main process:
    1. Collects observations by running episodes with current policy
    2. Sends actions back (or pre-generates them)
    3. Workers run full episodes and return transitions

    Since our auction has variable-length episodes and complex
    observation-action interleaving, we use a simpler approach:
    - Main process runs inference to get actions for ALL steps upfront
      using a "shadow" env to generate obs sequences
    - Workers execute the episodes with those pre-computed actions
    - This avoids IPC overhead on every step
    """

    def __init__(self, num_envs: int = 8, seed: int = 0):
        self.num_envs = num_envs
        self.base_seed = seed

    def run_parallel_episodes(
        self,
        get_action_fn: Callable,  # fn(obs, global_state) → (action, log_prob, value)
        n_episodes:    int,
        seed_start:    int = 0,
    ) -> List[dict]:
        """
        Run n_episodes in parallel batches of num_envs.
        Returns list of episode result dicts.
        
        Uses a two-phase approach:
        1. Run inference phase: use shadow envs to collect obs and get actions
        2. Replay phase: workers replay the episode with those actions
        
        For simplicity and correctness, we use ProcessPoolExecutor
        to run complete episodes in parallel with the policy baked in.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import functools

        results = []
        # Run in batches
        for batch_start in range(0, n_episodes, self.num_envs):
            batch_seeds = list(range(
                seed_start + batch_start,
                seed_start + min(batch_start + self.num_envs, n_episodes)
            ))
            # Run batch in parallel
            batch_results = self._run_batch(get_action_fn, batch_seeds)
            results.extend(batch_results)

        return results

    def _run_batch(self, get_action_fn, seeds: List[int]) -> List[dict]:
        """Run one batch of episodes in parallel using multiprocessing."""
        with mp.Pool(processes=len(seeds)) as pool:
            worker_fn = functools.partial(_run_single_episode, get_action_fn=get_action_fn)
            results = pool.map(worker_fn, seeds)
        return results


def _run_single_episode(seed: int, get_action_fn=None) -> dict:
    """
    Run one complete episode. Designed to be called in a subprocess.
    get_action_fn is passed as a picklable function.
    """
    env      = IPLAuctionEnv(retentions_enabled=True)
    obs_dict, _ = env.reset(seed=seed)
    step_data   = {t: [] for t in ALL_TEAM_IDS}
    steps       = 0

    while env.agents and steps < 30000:
        agent_id = env.agent_selection
        obs, reward, term, trunc, _ = env.last()

        if term or trunc:
            if step_data[agent_id]:
                step_data[agent_id][-1]["done"]   = True
                step_data[agent_id][-1]["reward"] += reward
            env.step(None)
            steps += 1
            continue

        global_state = env.state()

        if get_action_fn is not None:
            action, log_prob, value = get_action_fn(obs, global_state)
        else:
            action, log_prob, value = int(np.random.random() < 0.3), -0.7, 0.0

        step_data[agent_id].append({
            "obs":          obs.copy(),
            "global_state": global_state.copy(),
            "action":       action,
            "log_prob":     log_prob,
            "value":        value,
            "reward":       reward,
            "done":         False,
        })

        env.step(action)
        steps += 1

    scores = env.get_final_scores()
    state  = env.get_state_dict()

    for team_id in ALL_TEAM_IDS:
        final_reward = env._cumulative_rewards.get(team_id, 0.0)
        if step_data[team_id]:
            step_data[team_id][-1]["reward"] += final_reward

    mean_score = np.mean([s.get("final_score", 0) for s in scores.values()]) if scores else 0

    return {
        "seed":       seed,
        "step_data":  step_data,
        "scores":     scores,
        "steps":      steps,
        "mean_score": mean_score,
    }
