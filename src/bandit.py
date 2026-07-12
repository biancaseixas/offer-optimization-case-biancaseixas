"""Contextual Thompson Sampling with online Bayesian logistic regression per arm.

Implements the diagonal Laplace-approximation algorithm from Chapelle & Li (2011),
"An Empirical Evaluation of Thompson Sampling" (Algorithm 3): each arm keeps an
independent Gaussian posterior over its weight vector (mean `m`, diagonal
precision `q`). Selecting an arm samples weights from each arm's posterior and
scores by `P(conversion) * value(arm)` - the bandit analogue of `pCTR x bid` in
an ad auction. Updating an arm after an observed outcome takes a few Newton
steps from the current mean to find the new mode, then refreshes the diagonal
precision with the Laplace-approximation increment.

This module only deals with numpy arrays - it is agnostic of how the context
vector `x` was built/encoded (that preprocessing - scaling, one-hot encoding,
adding the bias term - happens in the notebook).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class BayesianLogisticArm:
    """Online Bayesian logistic regression for a single bandit arm.

    Posterior over weights w ~ N(m, diag(1/q)), q initialized to `prior_precision`
    (higher = stronger regularization / more confident prior at zero).
    """

    def __init__(self, n_features: int, prior_precision: float = 1.0):
        self.n_features = n_features
        self.m = np.zeros(n_features)
        self.q = np.full(n_features, prior_precision)

    def sample_weights(self, rng: np.random.Generator) -> np.ndarray:
        std = 1.0 / np.sqrt(self.q)
        return rng.normal(self.m, std)

    def predict_proba(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        w = self.m if weights is None else weights
        return float(sigmoid(np.dot(w, x)))

    def update(self, x: np.ndarray, y: int, n_newton_steps: int = 3, max_step: float = 2.0) -> None:
        """Single-example online update (Chapelle & Li, 2011, Sec. 3, Algorithm 3).

        Finds the new posterior mode via a few Newton steps from the current mean,
        then adds this example's contribution to the diagonal precision.

        `max_step` clips each Newton step: with only a handful of updates for an
        arm (early on, or an arm the bandit rarely picks), a single surprising
        example can otherwise send the Hessian's `p*(1-p)` term toward zero
        (quasi-separation, the classic logistic-regression divergence failure
        mode) and blow `w` up towards +-inf - which then makes that arm look
        artificially, permanently certain regardless of context. Clipping the
        step keeps each update bounded while still converging over many examples.
        """
        w = self.m.copy()
        for _ in range(n_newton_steps):
            p = sigmoid(np.dot(w, x))
            grad = self.q * (w - self.m) + (p - y) * x
            hess_diag = self.q + (x ** 2) * p * (1 - p)
            step = np.clip(grad / hess_diag, -max_step, max_step)
            w = w - step

        p_final = sigmoid(np.dot(w, x))
        self.q = self.q + (x ** 2) * p_final * (1 - p_final)
        self.m = w


class ContextualThompsonSampling:
    """Contextual Thompson Sampling bandit: one BayesianLogisticArm per offer.

    `arm_values` maps arm_id -> net value of a success for that arm (e.g. observed
    average ticket minus the offer's discount cost) - known upfront from the data,
    not learned. The bandit only learns P(conversion | context, arm); ranking uses
    sampled_P(conversion) * value(arm), mirroring pCTR x bid in ad auctions.
    """

    def __init__(self, arm_ids: List[str], n_features: int, arm_values: Dict[str, float],
                 prior_precision: float = 1.0, random_state: int = 42):
        self.arm_ids = list(arm_ids)
        self.arm_values = arm_values
        self.arms = {a: BayesianLogisticArm(n_features, prior_precision) for a in self.arm_ids}
        self.rng = np.random.default_rng(random_state)

    def select_arm(self, x: np.ndarray, candidate_arms: Optional[List[str]] = None) -> Tuple[str, Dict[str, float]]:
        """Samples weights per candidate arm, scores by sampled_P(conversion) * value(arm)."""
        candidates = candidate_arms if candidate_arms is not None else self.arm_ids
        scores = {}
        for arm_id in candidates:
            arm = self.arms[arm_id]
            w = arm.sample_weights(self.rng)
            p = arm.predict_proba(x, weights=w)
            scores[arm_id] = p * self.arm_values[arm_id]

        best_arm = max(scores, key=scores.get)
        return best_arm, scores

    def update(self, arm_id: str, x: np.ndarray, y: int) -> None:
        self.arms[arm_id].update(x, y)

    def predict_proba(self, arm_id: str, x: np.ndarray) -> float:
        """MAP probability estimate (uses the posterior mean, not a sample) - used
        for reporting/diagnostics, not for the exploration decision itself."""
        return self.arms[arm_id].predict_proba(x)
