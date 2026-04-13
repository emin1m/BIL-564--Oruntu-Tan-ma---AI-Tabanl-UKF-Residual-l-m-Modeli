"""
Hybrid Physics + Learned Residual Measurement for Continuous Sensors
UKF Tracking with a Neural-Corrected Measurement Model (No Particle Filter)

What this script does (end-to-end):
1) Keeps your physics: a_i(x)=sqrt(P0/(1+d^2)) on a 5x5 sensor grid
2) Uses a mismatched continuous sensor response to generate training + test data:
   - per-sensor gain g_i
   - per-sensor bias b_i
   - mild position-dependent shadowing term
3) Trains a small neural net g_theta to learn residual correction in measurement space:
      z_i(x) = a_i(x) + g_theta(phi_i(x))
4) Runs UKF where hx(x) returns continuous sensor amplitudes, and z is continuous
5) Compares:
   - UKF with physics-only measurements
   - UKF with physics + learned residual measurements (AI)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from trajectory_scenarios import (
    TrajectoryRolloutConfig,
    build_configured_trajectory_scenario,
    create_trajectory_scenario,
    default_filter_initial_state_for_trajectory,
    default_initial_state_for_trajectory,
    default_tracking_steps_for_trajectory,
    simulate_trajectory_and_measurements as simulate_trajectory_rollout,
    constant_velocity_transition,
    white_accel_process_noise,
)

# -----------------------------
# Optional: PyTorch (required for "AI residual")
# -----------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("This script requires PyTorch. Install it, then rerun.") from e

def sensor_coords_t(device: torch.device) -> torch.Tensor:
    return torch.tensor(sensor_coords, dtype=torch.float32, device=device)


# ============================================================
# 0) Sensor grid + physics (configurable)
# ============================================================
P0 = 1e4
num_sensors_x = 5
num_sensors_y = 5
num_sensors = num_sensors_x * num_sensors_y
rng_values = np.linspace(-30, 30, num_sensors_x)
sensor_coords = np.array([[x, y] for x in rng_values for y in rng_values], dtype=np.float64)  # (25,2)


def configure_sensor_grid_and_physics(
    p0: float = 1e4,
    sensors_x: int = 5,
    sensors_y: int = 5,
    coord_min: float = -30.0,
    coord_max: float = 30.0,
) -> None:
    """
    Global sensor/physics config. Keeping this configurable allows putting
    all experiment-driving parameters in main.
    """
    global P0, num_sensors_x, num_sensors_y, num_sensors, rng_values, sensor_coords

    P0 = float(p0)
    num_sensors_x = int(sensors_x)
    num_sensors_y = int(sensors_y)
    num_sensors = num_sensors_x * num_sensors_y
    rng_values = np.linspace(coord_min, coord_max, num_sensors_x)
    sensor_coords = np.array([[x, y] for x in rng_values for y in rng_values], dtype=np.float64)


def amplitude_vec_np(x_t: np.ndarray) -> np.ndarray:
    """
    Physics amplitude model:
      a_i(x) = sqrt(P0 / (1 + d^2)), d^2 = (xi-x)^2 + (yi-y)^2
    x_t: shape (4,)
    returns a: shape (25,)
    """
    x_t = np.asarray(x_t, dtype=np.float64).reshape(4)
    x, y = x_t[0], x_t[1]
    a = np.zeros(num_sensors, dtype=np.float64)
    for i, (xi, yi) in enumerate(sensor_coords):
        d2 = (xi - x) ** 2 + (yi - y) ** 2
        a[i] = sqrt(P0 / (1.0 + d2))
    return a


# ============================================================
# 1) "True" continuous sensor response for simulation (with mismatch)
#    (This creates the gap AI should learn)
# ============================================================
def make_true_sensor_params(
    seed: int = 123,
    gain_std: float = 0.08,
    bias_std: float = 0.8,
):
    rng = np.random.default_rng(seed)
    g_i = 1.0 + rng.normal(0.0, gain_std, size=num_sensors) # sensör kazançları, ortalama 1.0, sapma gain_std
    b_i = rng.normal(0.0, bias_std, size=num_sensors) # sensör biasları, ortalama 0.0, sapma bias_std
    return g_i.astype(np.float64), b_i.astype(np.float64)


shadow_sin_amp = 0.6
shadow_sin_freq = 0.12
shadow_cos_amp = 0.4
shadow_cos_freq = 0.10


def configure_shadowing(
    sin_amp: float = 0.6,
    sin_freq: float = 0.12,
    cos_amp: float = 0.4,
    cos_freq: float = 0.10,
) -> None:
    global shadow_sin_amp, shadow_sin_freq, shadow_cos_amp, shadow_cos_freq
    shadow_sin_amp = float(sin_amp)
    shadow_sin_freq = float(sin_freq)
    shadow_cos_amp = float(cos_amp)
    shadow_cos_freq = float(cos_freq)


def shadowing_term_np(x_t: np.ndarray) -> float:
    """
    Simple position-dependent mismatch (shadowing/multipath proxy).
    You can change this; keep it smooth so learning is feasible.
    """
    x, y = float(x_t[0]), float(x_t[1])
    return shadow_sin_amp * np.sin(shadow_sin_freq * x) + shadow_cos_amp * np.cos(shadow_cos_freq * y)


def true_measurement_np(
    x_t: np.ndarray,
    g_i: np.ndarray,
    b_i: np.ndarray,
) -> np.ndarray:
    """
    True (mismatched) continuous response:
      z_i(x) = g_i * a_i(x) + b_i + shadow(x)
    """
    a = amplitude_vec_np(x_t)
    z = g_i * a + b_i + shadowing_term_np(x_t)
    return z.astype(np.float64)


def sample_measurement_np(
    x_t: np.ndarray,
    g_i: np.ndarray,
    b_i: np.ndarray,
    meas_std: float = 2,
    seed: int | None = None,
) -> np.ndarray:
    """
    Continuous measurement:
      z_i ~ N(true_measurement_i(x), meas_std^2)
    """
    rng = np.random.default_rng(seed)
    z = true_measurement_np(x_t, g_i, b_i)
    if meas_std > 0.0:
        z = z + rng.normal(0.0, meas_std, size=num_sensors)
    return z


# ============================================================
# 2) Motion model (same CV as your setup)
# ============================================================
def F_constant_velocity(dt: float) -> np.ndarray:
    return constant_velocity_transition(dt)


def Q_white_accel(dt: float, tau: float) -> np.ndarray:
    return white_accel_process_noise(dt, tau)


def fx_cv(x: np.ndarray, dt: float) -> np.ndarray:
    return F_constant_velocity(dt) @ np.asarray(x, dtype=np.float64).reshape(4)


# ============================================================
# 3) UKF (sigma points + UT) with adaptive R hook
# ============================================================
def _force_symmetry(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _safe_cholesky(P: np.ndarray, max_tries: int = 8, eps: float = 1e-10) -> np.ndarray:
    P = _force_symmetry(P)
    I = np.eye(P.shape[0], dtype=P.dtype)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(P + jitter * I)
        except np.linalg.LinAlgError:
            jitter = eps if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("Cholesky failed: covariance not PD.")


def _safe_solve(A: np.ndarray, B: np.ndarray, max_tries: int = 8, eps: float = 1e-10) -> np.ndarray:
    A = _force_symmetry(A)
    I = np.eye(A.shape[0], dtype=A.dtype)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            return np.linalg.solve(A + jitter * I, B)
        except np.linalg.LinAlgError:
            jitter = eps if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("Solve failed: innovation covariance singular.")


class MerweSigmaPoints:
    def __init__(self, n: int, alpha: float = 0.3, beta: float = 2.0, kappa: float = 0.0):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lmbda = alpha**2 * (n + kappa) - n
        self.c = n + self.lmbda

        self.Wm = np.full(2 * n + 1, 0.5 / self.c, dtype=np.float64)
        self.Wc = np.full(2 * n + 1, 0.5 / self.c, dtype=np.float64)
        self.Wm[0] = self.lmbda / self.c
        self.Wc[0] = self.lmbda / self.c + (1.0 - alpha**2 + beta)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(self.n)
        P = np.asarray(P, dtype=np.float64).reshape(self.n, self.n)
        U = _safe_cholesky(self.c * P)

        sigmas = np.zeros((2 * self.n + 1, self.n), dtype=np.float64)
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i + 1] = x + U[:, i]
            sigmas[self.n + i + 1] = x - U[:, i]
        return sigmas


def unscented_transform(sigmas: np.ndarray, Wm: np.ndarray, Wc: np.ndarray, noise_cov: np.ndarray):
    mean = Wm @ sigmas
    d = sigmas.shape[1]
    cov = np.zeros((d, d), dtype=np.float64)
    for i in range(sigmas.shape[0]):
        e = (sigmas[i] - mean).reshape(-1, 1)
        cov += Wc[i] * (e @ e.T)
    cov += noise_cov
    cov = _force_symmetry(cov)
    return mean, cov


class UKF:
    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        fx,
        hx,
        dt: float,
        Q: np.ndarray,
        R_init: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        alpha: float = 0.3,
        beta: float = 2.0,
        kappa: float = 0.0,
        R_adapt_func=None,  # callable(z_pred)->R
    ):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.hx = hx
        self.dt = float(dt)

        self.Q = np.asarray(Q, dtype=np.float64).reshape(dim_x, dim_x)
        self.R = np.asarray(R_init, dtype=np.float64).reshape(dim_z, dim_z)

        self.x = np.asarray(x0, dtype=np.float64).reshape(dim_x)
        self.P = np.asarray(P0, dtype=np.float64).reshape(dim_x, dim_x)

        self.points = MerweSigmaPoints(dim_x, alpha=alpha, beta=beta, kappa=kappa)
        self.Wm = self.points.Wm
        self.Wc = self.points.Wc

        self.sigmas_f = None
        self.R_adapt_func = R_adapt_func

    def predict(self):
        sigmas = self.points.sigma_points(self.x, self.P)
        sigmas_f = np.zeros_like(sigmas)
        for i in range(sigmas.shape[0]):
            sigmas_f[i] = np.asarray(self.fx(sigmas[i], self.dt), dtype=np.float64).reshape(self.dim_x)

        self.sigmas_f = sigmas_f
        self.x, self.P = unscented_transform(sigmas_f, self.Wm, self.Wc, self.Q)
        self.P = _force_symmetry(self.P)

    def update(self, z: np.ndarray):
        if self.sigmas_f is None:
            raise RuntimeError("predict() must be called before update().")

        z = np.asarray(z, dtype=np.float64).reshape(self.dim_z)

        n_sig = self.sigmas_f.shape[0]
        sigmas_h = np.zeros((n_sig, self.dim_z), dtype=np.float64)
        for i in range(n_sig):
            sigmas_h[i] = np.asarray(self.hx(self.sigmas_f[i]), dtype=np.float64).reshape(self.dim_z)

        z_pred, S = unscented_transform(sigmas_h, self.Wm, self.Wc, self.R)

        if self.R_adapt_func is not None:
            self.R = np.asarray(self.R_adapt_func(z_pred), dtype=np.float64).reshape(self.dim_z, self.dim_z)
            z_pred, S = unscented_transform(sigmas_h, self.Wm, self.Wc, self.R)

        Pxz = np.zeros((self.dim_x, self.dim_z), dtype=np.float64)
        for i in range(n_sig):
            dx = (self.sigmas_f[i] - self.x).reshape(-1, 1)
            dz = (sigmas_h[i] - z_pred).reshape(-1, 1)
            Pxz += self.Wc[i] * (dx @ dz.T)

        S = _force_symmetry(S)
        K = _safe_solve(S, Pxz.T).T

        y = z - z_pred
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T
        self.P = _force_symmetry(self.P)

    def step(self, z: np.ndarray):
        self.predict()
        self.update(z)
        return self.x.copy(), self.P.copy()


# ============================================================
# 4) Neural residual measurement model g_theta(phi)
#    Shared per-sensor MLP: phi_i(x) -> residual_measurement_i
# ============================================================
class ResidualPerSensorMLP(nn.Module):
    """
    Input per sensor: [a_i, d2_i, x, y, vx, vy, xi, yi] -> residual measurement
    Shared weights across sensors, outputs 25 residual values for a batch.
    """

    def __init__(self, in_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, N, in_dim) -> residual: (B, N)
        B, N, D = feats.shape
        out = self.net(feats.view(B * N, D))
        return out.view(B, N)




def amplitude_and_d2_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B,4) on some device
    returns:
      a:  (B,25)
      d2: (B,25)
    """
    B = x.shape[0]
    sc = sensor_coords_t(x.device).unsqueeze(0).expand(B, -1, -1)  # (B,25,2)
    xy = x[:, 0:2].unsqueeze(1)  # (B,1,2)
    diff = sc - xy
    d2 = (diff ** 2).sum(dim=-1)  # (B,25)
    a = torch.sqrt(torch.tensor(P0, dtype=torch.float32, device=x.device) / (1.0 + d2))
    return a, d2

def hx_meas_physics_only(x_np: np.ndarray) -> np.ndarray:
    return amplitude_vec_np(x_np).astype(np.float64)


@torch.no_grad()
def hx_meas_hybrid(
    x_np: np.ndarray,
    net: nn.Module,
    device: str = "cpu",
) -> np.ndarray:
    """
    Hybrid continuous measurement:
      z_hat(x) = a(x) + residual_net(phi)
    returns (25,) numpy
    """
    x = torch.tensor(np.asarray(x_np, dtype=np.float32).reshape(1, 4), device=device)
    a, d2 = amplitude_and_d2_torch(x)  # (1,25), (1,25)

    sc = sensor_coords_t(x.device).unsqueeze(0)
    # build features per sensor: [a_i, d2_i, x,y,vx,vy, xi, yi]
    x_rep = x.unsqueeze(1).expand(-1, num_sensors, -1)  # (1,25,4)
    feats = torch.cat(
        [
            a.unsqueeze(-1),          # (1,25,1)
            d2.unsqueeze(-1),         # (1,25,1)
            x_rep,                    # (1,25,4)
            sc,                       # (1,25,2)
        ],
        dim=-1,
    )  # (1,25,8)

    residual = net(feats).squeeze(0)  # (25,)
    z_pred = a.squeeze(0) + residual
    return z_pred.detach().cpu().numpy().astype(np.float64)


# ============================================================
# 5) Training data generation
# ============================================================
def sample_random_states(
    N: int,
    roi_min: float = -30.0,
    roi_max: float = 30.0,
    vmin: float = -2.0,
    vmax: float = 2.0,
    seed: int = 202,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(roi_min, roi_max, size=(N, 1))
    ys = rng.uniform(roi_min, roi_max, size=(N, 1))
    vxs = rng.uniform(vmin, vmax, size=(N, 1))
    vys = rng.uniform(vmin, vmax, size=(N, 1))
    return np.hstack([xs, ys, vxs, vys]).astype(np.float64)


def build_dataset(
    N: int,
    g_i: np.ndarray,
    b_i: np.ndarray,
    meas_std: float = 0.5,
    seed: int = 303,
    roi_min: float = -30.0,
    roi_max: float = 30.0,
    vmin: float = -2.0,
    vmax: float = 2.0,
    state_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if state_seed is None:
        state_seed = seed
    X = sample_random_states(
        N,
        roi_min=roi_min,
        roi_max=roi_max,
        vmin=vmin,
        vmax=vmax,
        seed=state_seed,
    )
    Z = np.zeros((N, num_sensors), dtype=np.float64)
    for i in range(N):
        Z[i] = sample_measurement_np(X[i], g_i, b_i, meas_std=meas_std, seed=int(rng.integers(1, 1_000_000_000)))
    return X, Z


# ============================================================
# 6) Train residual net (physics + residual, MSELoss)
# ============================================================
def train_residual_net(
    X_train: np.ndarray,
    Z_train: np.ndarray,
    X_val: np.ndarray,
    Z_val: np.ndarray,
    device: str = "cpu",
    epochs: int = 12,
    batch_size: int = 512,
    lr: float = 3e-4,
    model_hidden: int = 128,
) -> nn.Module:
    net = ResidualPerSensorMLP(in_dim=8, hidden=model_hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    Xtr = torch.tensor(X_train.astype(np.float32), device=device)
    Ztr = torch.tensor(Z_train.astype(np.float32), device=device)
    Xva = torch.tensor(X_val.astype(np.float32), device=device)
    Zva = torch.tensor(Z_val.astype(np.float32), device=device)

    def make_predictions(x_batch: torch.Tensor) -> torch.Tensor:
        # x_batch: (B,4)
        a, d2 = amplitude_and_d2_torch(x_batch)  # (B,25)
        sc = sensor_coords_t(x_batch.device).unsqueeze(0).expand(x_batch.shape[0], -1, -1)
        x_rep = x_batch.unsqueeze(1).expand(-1, num_sensors, -1)  # (B,25,4)
        feats = torch.cat([a.unsqueeze(-1), d2.unsqueeze(-1), x_rep, sc], dim=-1)  # (B,25,8)

        residual = net(feats)  # (B,25)
        return a + residual  # (B,25)

    N = Xtr.shape[0]
    idx = torch.arange(N, device=device)

    for ep in range(1, epochs + 1):
        net.train()
        perm = idx[torch.randperm(N, device=device)]
        total_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            b = perm[start : start + batch_size]
            xb = Xtr[b]
            zb = Ztr[b]

            z_hat = make_predictions(xb)
            loss = F.mse_loss(z_hat, zb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        # validation
        net.eval()
        with torch.no_grad():
            z_hat_val = make_predictions(Xva)
            val_loss = F.mse_loss(z_hat_val, Zva).item()

        print(f"Epoch {ep:02d} | train MSE: {total_loss/max(n_batches,1):.4f} | val MSE: {val_loss:.4f}")

    return net


# ============================================================
# 7) Tracking experiment: compare physics-only vs hybrid (AI) UKF
# ============================================================
def simulate_trajectory_and_measurements(
    T: int,
    dt: float,
    x0_true: np.ndarray,
    Q: np.ndarray,
    g_i: np.ndarray,
    b_i: np.ndarray,
    meas_std: float = 0.5,
    seed: int = 777,
    trajectory_scenario=None,
) -> tuple[np.ndarray, np.ndarray]:
    scenario = trajectory_scenario
    if scenario is None:
        scenario = create_trajectory_scenario("cv_gaussian", dt=dt, process_noise_cov=Q)

    rollout = TrajectoryRolloutConfig(num_steps=T, x0_true=x0_true, seed=seed)

    def measurement_sampler(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return sample_measurement_np(
            x,
            g_i,
            b_i,
            meas_std=meas_std,
            seed=int(rng.integers(1, 1_000_000_000)),
        )

    return simulate_trajectory_rollout(
        scenario=scenario,
        rollout=rollout,
        measurement_dim=num_sensors,
        measurement_sampler=measurement_sampler,
    )


def run_ukf_tracking(
    xs_true: np.ndarray,
    zs_meas: np.ndarray,
    dt: float,
    Q: np.ndarray,
    x0_est: np.ndarray,
    P0: np.ndarray,
    hx_callable,  # hx(x)->predicted continuous measurements
    meas_std: float = 0.5,
    meas_std_floor: float = 1e-3,
    ukf_alpha: float = 0.3,
    ukf_beta: float = 2.0,
    ukf_kappa: float = 0.0,
    R_adapt_func=None,
) -> np.ndarray:
    R0 = np.eye(num_sensors, dtype=np.float64) * (max(meas_std, meas_std_floor) ** 2)

    ukf = UKF(
        dim_x=4,
        dim_z=num_sensors,
        fx=fx_cv,
        hx=hx_callable,
        dt=dt,
        Q=Q,
        R_init=R0,
        x0=x0_est,
        P0=P0,
        alpha=ukf_alpha,
        beta=ukf_beta,
        kappa=ukf_kappa,
        R_adapt_func=R_adapt_func,
    )

    T = zs_meas.shape[0]
    xs_est = np.zeros((T, 4), dtype=np.float64)
    for k in range(T):
        xk, _Pk = ukf.step(zs_meas[k])
        xs_est[k] = xk
    return xs_est


def rmse_pos(xs_est: np.ndarray, xs_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((xs_est[:, :2] - xs_true[:, :2]) ** 2)))


# ============================================================
# 8) Main: train + test
# ============================================================
if __name__ == "__main__":
    # -----------------------------
    # 0) Reproducibility (sonucu etkiler)
    # -----------------------------
    seed_numpy = 0               # numpy random seed
    seed_torch = 0               # torch random seed
    seed_sensor_params = 123     # gain/bias mismatch seed
    seed_dataset_noise = 303     # train/val measurement noise seed
    seed_dataset_states = 303    # train/val state sampling seed
    seed_tracking = 777          # tracking trajectory + noise seed

    np.random.seed(seed_numpy)
    torch.manual_seed(seed_torch)

    # -----------------------------
    # 1) Sensor + physics config (sonucu etkiler)
    # -----------------------------
    sensor_p0 = 1e4              # fizik modelindeki güç sabiti
    sensor_grid_x = 5            # x eksenindeki sensör sayısı
    sensor_grid_y = 5            # y eksenindeki sensör sayısı
    sensor_coord_min = -30.0     # sensör grid alt sınırı
    sensor_coord_max = 30.0      # sensör grid üst sınırı

    configure_sensor_grid_and_physics(
        p0=sensor_p0,
        sensors_x=sensor_grid_x,
        sensors_y=sensor_grid_y,
        coord_min=sensor_coord_min,
        coord_max=sensor_coord_max,
    )

    # -----------------------------
    # 2) True mismatch / environment config (sonucu etkiler)
    # -----------------------------
    gain_std = 0.10              # sensör kazanç sapması std
    bias_std = 0.8               # sensör bias sapması std
    meas_std = 1              # ölçüm gürültüsü std (R ile ilişkili) 0 yapınca physics only ukf çok kötü oluyor, nedenini anla.

    # shadowing: z = g*a + b + A_sin*sin(f_sin*x) + A_cos*cos(f_cos*y)
    shadow_sin_amplitude = 0.6   # sin bileşeni genliği
    shadow_sin_frequency = 0.12  # sin bileşeni frekansı
    shadow_cos_amplitude = 0.4   # cos bileşeni genliği
    shadow_cos_frequency = 0.10  # cos bileşeni frekansı

    configure_shadowing(
        sin_amp=shadow_sin_amplitude,
        sin_freq=shadow_sin_frequency,
        cos_amp=shadow_cos_amplitude,
        cos_freq=shadow_cos_frequency,
    )

    # -----------------------------
    # 3) Train/val dataset config (sonucu etkiler)
    # -----------------------------
    roi_min = -30.0              # durum örnekleme alanı alt sınır
    roi_max = 30.0               # durum örnekleme alanı üst sınır
    vmin = -2.0                  # hız alt sınırı
    vmax = 2.0                   # hız üst sınırı
    n_train = 30000              # train örnek sayısı
    n_val = 8000                 # validation örnek sayısı

    # -----------------------------
    # 4) Residual NN train config (sonucu etkiler)
    # -----------------------------
    train_epochs = 200           # epoch sayısı
    batch_size = 512             # batch boyutu
    lr = 3e-4                    # optimizer learning rate
    model_hidden = 128           # residual MLP hidden katman genişliği

    # -----------------------------
    # 5) Tracking / motion config (sonucu etkiler)
    # -----------------------------
    dt = 0.25                    # örnekleme aralığı
    tau = 1e-2                   # white-accel process noise ölçeği (Q)
    t_steps = 80                 # tracking adım sayısı
    x0_true = np.array([-8.0, 10.0, 1.2, -0.8], dtype=np.float64)   # gerçek başlangıç durumu
    x0_est = np.array([-6.5, 9.0, 0.6, -0.4], dtype=np.float64)     # filtre başlangıç tahmini
    p_init = np.diag([4.0, 4.0, 1.0, 1.0]).astype(np.float64)       # filtre başlangıç kovaryansı

    trajectory_kind = "cv_gaussian_extended"  # secenekler: cv_gaussian, cv_gaussian_extended
    q_scale = 1.0
    extended_process_noise_scale = 1.6
    extended_motion_bias = np.array([0.0, 0.0, -0.010, 0.006], dtype=np.float64)
    t_steps = default_tracking_steps_for_trajectory(trajectory_kind, base_steps=t_steps, extended_steps=110)
    x0_true = default_initial_state_for_trajectory(
        trajectory_kind=trajectory_kind,
        cv_x0_true=x0_true,
    )
    x0_est = default_filter_initial_state_for_trajectory(
        trajectory_kind=trajectory_kind,
        cv_x0_est=x0_est,
    )

    # -----------------------------
    # 6) UKF hyper-parameters (sonucu etkiler)
    # -----------------------------
    ukf_alpha = 0.3              # sigma-point spread
    ukf_beta = 2.0               # Gaussian için genelde 2
    ukf_kappa = 0.0              # sigma-point secondary scaling
    meas_std_floor = 1e-3        # R hesabında std alt sınırı
    r_adapt_func = None          # adaptif R fonksiyonu (yoksa None)

    # -----------------------------
    # Device
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # True continuous sensor mismatch params
    g_i_true, b_i_true = make_true_sensor_params(
        seed=seed_sensor_params,
        gain_std=gain_std,
        bias_std=bias_std,
    )
    print("True continuous sensor mismatch set.")

    # --------
    # Train data
    # --------
    X_all, Z_all = build_dataset(
        n_train + n_val,
        g_i_true,
        b_i_true,
        meas_std=meas_std,
        seed=seed_dataset_noise,
        roi_min=roi_min,
        roi_max=roi_max,
        vmin=vmin,
        vmax=vmax,
        state_seed=seed_dataset_states,
    )

    X_train, Z_train = X_all[:n_train], Z_all[:n_train]
    X_val, Z_val = X_all[n_train:], Z_all[n_train:]

    print("Training residual measurement net...")
    net = train_residual_net(
        X_train,
        Z_train,
        X_val,
        Z_val,
        device=device,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
        model_hidden=model_hidden,
    )

    # --------
    # Tracking test
    # --------
    Q = q_scale * Q_white_accel(dt, tau)
    print("Q scale:", q_scale)
    trajectory_scenario = build_configured_trajectory_scenario(
        trajectory_kind=trajectory_kind,
        dt=dt,
        cv_process_noise_cov=Q,
        extended_process_noise_scale=extended_process_noise_scale,
        extended_motion_bias=extended_motion_bias,
    )
    print("Trajectory scenario:", trajectory_kind)

    xs_true, zs_meas = simulate_trajectory_and_measurements(
        T=t_steps,
        dt=dt,
        x0_true=x0_true,
        Q=Q,
        g_i=g_i_true,
        b_i=b_i_true,
        meas_std=meas_std,
        seed=seed_tracking,
        trajectory_scenario=trajectory_scenario,
    )

    # hx for physics-only UKF (returns amplitude measurement)
    def hx_phys(x_np: np.ndarray) -> np.ndarray:
        return hx_meas_physics_only(x_np)

    # hx for hybrid UKF (physics + residual net)
    def hx_hybrid(x_np: np.ndarray) -> np.ndarray:
        return hx_meas_hybrid(x_np, net=net, device=device)

    print("Running UKF (physics-only)...")
    xs_est_phys = run_ukf_tracking(
        xs_true,
        zs_meas,
        dt,
        Q,
        x0_est,
        p_init,
        hx_phys,
        meas_std=meas_std,
        meas_std_floor=meas_std_floor,
        ukf_alpha=ukf_alpha,
        ukf_beta=ukf_beta,
        ukf_kappa=ukf_kappa,
        R_adapt_func=r_adapt_func,
    )

    print("Running UKF (hybrid physics + AI residual)...")
    xs_est_hyb = run_ukf_tracking(
        xs_true,
        zs_meas,
        dt,
        Q,
        x0_est,
        p_init,
        hx_hybrid,
        meas_std=meas_std,
        meas_std_floor=meas_std_floor,
        ukf_alpha=ukf_alpha,
        ukf_beta=ukf_beta,
        ukf_kappa=ukf_kappa,
        R_adapt_func=r_adapt_func,
    )

    rmse_phys = rmse_pos(xs_est_phys, xs_true)
    rmse_hyb = rmse_pos(xs_est_hyb, xs_true)
    print(f"Position RMSE | physics-only UKF: {rmse_phys:.4f}")
    print(f"Position RMSE | hybrid UKF      : {rmse_hyb:.4f}")

    # --------
    # Plot
    # --------
    plt.figure(figsize=(10, 8))
    plt.plot(xs_true[:, 0], xs_true[:, 1], "r-", linewidth=2.0, label="True")
    plt.plot(xs_est_phys[:, 0], xs_est_phys[:, 1], "k--", linewidth=2.0, label="UKF (physics-only)")
    plt.plot(xs_est_hyb[:, 0], xs_est_hyb[:, 1], "g-", linewidth=2.0, label="UKF (hybrid physics+AI)")

    plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], s=80, marker="^", alpha=0.7, label="Sensors")

    plt.title("Continuous Sensor Tracking: UKF with Physics vs Hybrid Physics+AI")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: time-series error plot
    err_phys = np.linalg.norm(xs_est_phys[:, :2] - xs_true[:, :2], axis=1)
    err_hyb = np.linalg.norm(xs_est_hyb[:, :2] - xs_true[:, :2], axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(err_phys, "k--", label="pos error (physics-only)")
    plt.plot(err_hyb, "g-", label="pos error (hybrid)")
    plt.title("Position Error vs Time")
    plt.xlabel("t")
    plt.ylabel("||e_pos||")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
