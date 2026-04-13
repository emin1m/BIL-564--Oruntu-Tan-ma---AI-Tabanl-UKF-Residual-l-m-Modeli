from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajectory_scenarios import TrajectoryScenario, constant_velocity_transition, register_trajectory_type


def make_extended_gaussian_initial_state() -> np.ndarray:
    return np.array([22.0, -18.0, -1.55, 1.10], dtype=np.float64)


def make_extended_gaussian_initial_estimate() -> np.ndarray:
    return np.array([20.5, -19.5, -1.05, 0.70], dtype=np.float64)


def make_extended_gaussian_motion_bias() -> np.ndarray:
    # Keeps the model Gaussian while nudging velocity so the path differs from the default CV case.
    return np.array([0.0, 0.0, -0.010, 0.006], dtype=np.float64)


def make_extended_gaussian_motion_jitter_std() -> np.ndarray:
    # Very small random drift so the path is not overly regular.
    return np.array([0.0, 0.0, 0.0015, 0.0010], dtype=np.float64)


@dataclass
class ExtendedGaussianTrajectory(TrajectoryScenario):
    process_noise_cov: np.ndarray | None = None
    motion_bias: np.ndarray | None = None
    motion_jitter_std: np.ndarray | None = None
    name: str = "cv_gaussian_extended"

    def __post_init__(self) -> None:
        if self.process_noise_cov is None:
            self.process_noise_cov = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        else:
            self.process_noise_cov = np.asarray(self.process_noise_cov, dtype=np.float64).reshape(self.state_dim, self.state_dim)

        if self.motion_bias is None:
            self.motion_bias = make_extended_gaussian_motion_bias()
        else:
            self.motion_bias = np.asarray(self.motion_bias, dtype=np.float64).reshape(self.state_dim)

        if self.motion_jitter_std is None:
            self.motion_jitter_std = make_extended_gaussian_motion_jitter_std()
        else:
            self.motion_jitter_std = np.asarray(self.motion_jitter_std, dtype=np.float64).reshape(self.state_dim)

    def step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(self.state_dim)
        motion_jitter = rng.normal(0.0, self.motion_jitter_std, size=self.state_dim)
        x_next = constant_velocity_transition(self.dt) @ x + self.motion_bias + motion_jitter
        noise = rng.multivariate_normal(np.zeros(self.state_dim, dtype=np.float64), self.process_noise_cov)
        return x_next + noise


register_trajectory_type(
    "cv_gaussian_extended",
    lambda dt, process_noise_cov, motion_bias=None, motion_jitter_std=None, **kwargs: ExtendedGaussianTrajectory(
        dt=dt,
        process_noise_cov=process_noise_cov,
        motion_bias=motion_bias,
        motion_jitter_std=motion_jitter_std,
        **kwargs,
    ),
)
