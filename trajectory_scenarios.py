from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


MeasurementSampler = Callable[[np.ndarray, np.random.Generator], np.ndarray]
TrajectoryFactory = Callable[..., "TrajectoryScenario"]


def constant_velocity_transition(dt: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
def white_accel_process_noise(dt: float, tau: float) -> np.ndarray:
    dt2 = dt * dt
    dt3 = dt2 * dt
    Q = np.array(
        [
            [dt3 / 3.0, 0.0, dt2 / 2.0, 0.0],
            [0.0, dt3 / 3.0, 0.0, dt2 / 2.0],
            [dt2 / 2.0, 0.0, dt, 0.0],
            [0.0, dt2 / 2.0, 0.0, dt],
        ],
        dtype=np.float64,
    )
    return tau * Q


def build_process_noise_from_accel_distribution(
    dt: float,
    accel_std_x: float,
    accel_std_y: float,
    accel_corr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    accel_cov = np.array(
        [
            [accel_std_x**2, accel_corr * accel_std_x * accel_std_y],
            [accel_corr * accel_std_x * accel_std_y, accel_std_y**2],
        ],
        dtype=np.float64,
    )
    dt2 = dt * dt
    G = np.array(
        [
            [0.5 * dt2, 0.0],
            [0.0, 0.5 * dt2],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=np.float64,
    )
    Q = G @ accel_cov @ G.T
    return accel_cov, Q


def _as_state_vector(x: np.ndarray, state_dim: int) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(state_dim).copy()


@dataclass(frozen=True)
class TrajectoryRolloutConfig:
    num_steps: int
    x0_true: np.ndarray
    seed: int = 777

    def initial_state(self, state_dim: int) -> np.ndarray:
        return _as_state_vector(self.x0_true, state_dim)


@dataclass
class TrajectoryScenario:
    dt: float
    state_dim: int = 4
    name: str = "base"

    def step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ConstantVelocityGaussianTrajectory(TrajectoryScenario):
    process_noise_cov: np.ndarray | None = None
    name: str = "cv_gaussian"

    def __post_init__(self) -> None:
        if self.process_noise_cov is None:
            self.process_noise_cov = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        else:
            self.process_noise_cov = np.asarray(self.process_noise_cov, dtype=np.float64).reshape(self.state_dim, self.state_dim)

    def step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x_next = constant_velocity_transition(self.dt) @ _as_state_vector(x, self.state_dim)
        noise = rng.multivariate_normal(np.zeros(self.state_dim, dtype=np.float64), self.process_noise_cov)
        return x_next + noise


TRAJECTORY_REGISTRY: dict[str, TrajectoryFactory] = {}


def register_trajectory_type(name: str, factory: TrajectoryFactory) -> None:
    TRAJECTORY_REGISTRY[name] = factory


def create_trajectory_scenario(kind: str, **kwargs) -> TrajectoryScenario:
    try:
        factory = TRAJECTORY_REGISTRY[kind]
    except KeyError as exc:
        available = ", ".join(sorted(TRAJECTORY_REGISTRY))
        raise ValueError(f"Unsupported trajectory kind '{kind}'. Available: {available}") from exc
    return factory(**kwargs)


def build_configured_trajectory_scenario(
    trajectory_kind: str,
    dt: float,
    cv_process_noise_cov: np.ndarray | None = None,
    extended_process_noise_scale: float = 1.6,
    extended_motion_bias: np.ndarray | None = None,
) -> TrajectoryScenario:
    if trajectory_kind == "cv_gaussian":
        if cv_process_noise_cov is None:
            raise ValueError("cv_process_noise_cov is required for 'cv_gaussian'.")
        return create_trajectory_scenario(
            "cv_gaussian",
            dt=dt,
            process_noise_cov=cv_process_noise_cov,
        )

    if trajectory_kind == "cv_gaussian_extended":
        if cv_process_noise_cov is None:
            raise ValueError("cv_process_noise_cov is required for 'cv_gaussian_extended'.")
        return create_trajectory_scenario(
            "cv_gaussian_extended",
            dt=dt,
            process_noise_cov=cv_process_noise_cov * float(extended_process_noise_scale),
            motion_bias=extended_motion_bias,
        )

    available = ", ".join(sorted(TRAJECTORY_REGISTRY))
    raise ValueError(f"Unsupported trajectory_kind '{trajectory_kind}'. Available: {available}")


def default_initial_state_for_trajectory(
    trajectory_kind: str,
    cv_x0_true: np.ndarray,
    extended_x0_true: np.ndarray | None = None,
) -> np.ndarray:
    if trajectory_kind == "cv_gaussian_extended":
        if extended_x0_true is None:
            from trajectory_scenarios_gaussian_extended import make_extended_gaussian_initial_state

            return make_extended_gaussian_initial_state()
        return np.asarray(extended_x0_true, dtype=np.float64).reshape(4).copy()

    return np.asarray(cv_x0_true, dtype=np.float64).reshape(4).copy()


def default_filter_initial_state_for_trajectory(
    trajectory_kind: str,
    cv_x0_est: np.ndarray,
    extended_x0_est: np.ndarray | None = None,
) -> np.ndarray:
    if trajectory_kind == "cv_gaussian_extended":
        if extended_x0_est is None:
            from trajectory_scenarios_gaussian_extended import make_extended_gaussian_initial_estimate

            return make_extended_gaussian_initial_estimate()
        return np.asarray(extended_x0_est, dtype=np.float64).reshape(4).copy()

    return np.asarray(cv_x0_est, dtype=np.float64).reshape(4).copy()


def default_tracking_steps_for_trajectory(
    trajectory_kind: str,
    base_steps: int,
    extended_steps: int = 110,
) -> int:
    if trajectory_kind == "cv_gaussian_extended":
        return int(extended_steps)
    return int(base_steps)


def simulate_trajectory_and_measurements(
    scenario: TrajectoryScenario,
    rollout: TrajectoryRolloutConfig,
    measurement_dim: int,
    measurement_sampler: MeasurementSampler,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rollout.seed)
    x = rollout.initial_state(scenario.state_dim)

    xs = np.zeros((rollout.num_steps, scenario.state_dim), dtype=np.float64)
    zs = np.zeros((rollout.num_steps, measurement_dim), dtype=np.float64)

    for k in range(rollout.num_steps):
        x = scenario.step(x, rng)
        xs[k] = x
        zs[k] = np.asarray(measurement_sampler(x, rng), dtype=np.float64).reshape(measurement_dim)

    return xs, zs


register_trajectory_type(
    "cv_gaussian",
    lambda dt, process_noise_cov, **kwargs: ConstantVelocityGaussianTrajectory(
        dt=dt,
        process_noise_cov=process_noise_cov,
        **kwargs,
    ),
)

# Import scenario extensions so they self-register in the shared registry.
import trajectory_scenarios_gaussian_extended  # noqa: E402,F401
