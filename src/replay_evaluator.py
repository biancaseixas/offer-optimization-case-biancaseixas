"""Offline evaluation for the contextual bandit: the Replay Method (Li et al.,
2011, "Unbiased Offline Evaluation of Contextual-Bandit-Based News Article
Recommendation Algorithms") against the real logged data, plus a synthetic
simulation environment for a clean cumulative-regret learning curve.

Replay Method premise (validated in notebook 1): offer assignment in the
historical log is close to uniformly random across the 10 offers, so simply
discarding rounds where the policy's chosen arm does not match the logged arm
gives an unbiased estimate of that policy's expected reward - no need to
estimate propensity scores.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class ReplayResult:
    policy_name: str
    n_total: int
    n_kept: int
    kept_conversion_rate: float
    kept_avg_value_per_contact: float
    cumulative_reward: np.ndarray  # over kept rounds, in kept-round order
    cumulative_value: np.ndarray


def run_replay(
    select_fn: Callable[[np.ndarray, int], str],
    update_fn: Optional[Callable[[str, np.ndarray, int], None]],
    X: np.ndarray,
    logged_arms: np.ndarray,
    rewards: np.ndarray,
    arm_values: Dict[str, float],
    policy_name: str,
) -> ReplayResult:
    """Li et al. (2011) replay method.

    Args:
        select_fn(x, row_index) -> arm_id chosen by the policy being evaluated.
        update_fn(arm_id, x, reward) -> called only on kept rounds, or None for
            policies that don't learn (random, static).
        X: (n, d) context matrix, rows already sorted in chronological order
            (by round_day, then a stable tiebreak).
        logged_arms: (n,) array of the offer actually sent historically.
        rewards: (n,) binary array (e.g. `completed`).
        arm_values: arm_id -> net value of a success, used to convert the binary
            reward into a $ value for the financial comparison.
    """
    n = X.shape[0]
    kept_rewards: List[int] = []
    kept_values: List[float] = []

    for i in range(n):
        x = X[i]
        chosen = select_fn(x, i)
        if chosen == logged_arms[i]:
            r = int(rewards[i])
            kept_rewards.append(r)
            kept_values.append(r * arm_values[chosen])
            if update_fn is not None:
                update_fn(chosen, x, r)

    n_kept = len(kept_rewards)
    kept_rewards_arr = np.array(kept_rewards, dtype=float)
    kept_values_arr = np.array(kept_values, dtype=float)

    return ReplayResult(
        policy_name=policy_name,
        n_total=n,
        n_kept=n_kept,
        kept_conversion_rate=float(kept_rewards_arr.mean()) if n_kept else float("nan"),
        kept_avg_value_per_contact=float(kept_values_arr.mean()) if n_kept else float("nan"),
        cumulative_reward=np.cumsum(kept_rewards_arr),
        cumulative_value=np.cumsum(kept_values_arr),
    )


def fit_environment_models(
    X: np.ndarray, arms: np.ndarray, rewards: np.ndarray, arm_ids: List[str]
) -> Dict[str, LogisticRegression]:
    """Fits one logistic-regression 'ground truth' response model per arm on all
    historical rows where that arm was actually sent - used as a synthetic
    environment for the simulation track (NOT used inside the bandit itself).
    """
    models = {}
    for arm_id in arm_ids:
        mask = arms == arm_id
        model = LogisticRegression(max_iter=2000)
        model.fit(X[mask], rewards[mask])
        models[arm_id] = model
    return models


def env_predict_proba(model: LogisticRegression, x: np.ndarray) -> float:
    return float(model.predict_proba(x.reshape(1, -1))[0, 1])


@dataclass
class SimulationResult:
    policy_name: str
    cumulative_reward: np.ndarray
    cumulative_regret: np.ndarray


def simulate_policy(
    select_fn: Callable[[np.ndarray], str],
    update_fn: Optional[Callable[[str, np.ndarray, int], None]],
    env_models: Dict[str, LogisticRegression],
    arm_values: Dict[str, float],
    context_pool: np.ndarray,
    n_rounds: int,
    rng: np.random.Generator,
    policy_name: str,
) -> SimulationResult:
    """Runs a policy for `n_rounds` simulated rounds against the fitted synthetic
    environment. Contexts are bootstrap-sampled (with replacement) from real
    customer context rows - this track is illustrative, not a measured fact.
    """
    n_pool = context_pool.shape[0]
    cum_reward = np.zeros(n_rounds)
    cum_regret = np.zeros(n_rounds)
    running_reward = 0.0
    running_regret = 0.0

    arm_ids = list(env_models.keys())

    for t in range(n_rounds):
        x = context_pool[rng.integers(0, n_pool)]

        true_probs = {a: env_predict_proba(env_models[a], x) for a in arm_ids}
        best_value = max(true_probs[a] * arm_values[a] for a in arm_ids)

        chosen = select_fn(x)
        chosen_true_value = true_probs[chosen] * arm_values[chosen]
        reward = 1 if rng.random() < true_probs[chosen] else 0

        if update_fn is not None:
            update_fn(chosen, x, reward)

        running_reward += reward * arm_values[chosen]
        running_regret += best_value - chosen_true_value
        cum_reward[t] = running_reward
        cum_regret[t] = running_regret

    return SimulationResult(policy_name=policy_name, cumulative_reward=cum_reward, cumulative_regret=cum_regret)


def best_static_arm(env_models: Dict[str, LogisticRegression], context_pool: np.ndarray, arm_values: Dict[str, float]) -> str:
    """The single 'champion' offer a no-personalization policy would always send -
    the arm with the highest average expected value across the customer base."""
    avg_value = {}
    for arm_id, model in env_models.items():
        probs = model.predict_proba(context_pool)[:, 1]
        avg_value[arm_id] = float(probs.mean()) * arm_values[arm_id]
    return max(avg_value, key=avg_value.get)
