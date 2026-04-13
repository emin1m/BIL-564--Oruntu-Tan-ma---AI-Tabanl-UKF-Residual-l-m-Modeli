"""
Hybrid Physics + Learned Residual Measurement for Continuous Sensors
UKF Tracking with a CNN-Corrected Measurement Model

This module keeps the same residual-learning formulation as
`hybrid_ukf_residual_tracking.py`, but replaces the per-sensor MLP with a
small CNN that operates on the sensor grid and predicts one residual value
per sensor.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("This script requires PyTorch. Install it, then rerun.") from e

import hybrid_ukf_residual_tracking as residual_mod


class ResidualGridCNN(nn.Module):
    """
    Input grid channels per sensor:
      [a_i, d2_i, x, y, vx, vy, xi, yi]
    Output:
      residual grid with one value per sensor location.
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feat_grid: torch.Tensor) -> torch.Tensor:
        # feat_grid: (B, C, H, W) -> residual: (B, H, W)
        return self.net(feat_grid).squeeze(1)


def residual_feature_grid_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build CNN features on the sensor grid.
    x: (B,4)
    returns:
      a_flat: (B,N)
      feat_grid: (B,8,H,W)
    """
    a, d2 = residual_mod.amplitude_and_d2_torch(x)
    batch_size = x.shape[0]
    grid_h = residual_mod.num_sensors_x
    grid_w = residual_mod.num_sensors_y

    sc = residual_mod.sensor_coords_t(x.device).unsqueeze(0).expand(batch_size, -1, -1)
    x_rep = x.unsqueeze(1).expand(-1, residual_mod.num_sensors, -1)
    feat_flat = torch.cat([a.unsqueeze(-1), d2.unsqueeze(-1), x_rep, sc], dim=-1)
    feat_grid = feat_flat.view(batch_size, grid_h, grid_w, 8).permute(0, 3, 1, 2).contiguous()
    return a, feat_grid


@torch.no_grad()
def hx_meas_hybrid_cnn(
    x_np: np.ndarray,
    net: nn.Module,
    device: str = "cpu",
) -> np.ndarray:
    """
    Hybrid continuous measurement:
      z_hat(x) = a(x) + residual_cnn(phi_grid)
    returns (N,) numpy
    """
    x = torch.tensor(np.asarray(x_np, dtype=np.float32).reshape(1, 4), device=device)
    a, feat_grid = residual_feature_grid_torch(x)
    residual_grid = net(feat_grid)
    residual = residual_grid.reshape(1, residual_mod.num_sensors)
    z_pred = a + residual
    return z_pred.squeeze(0).detach().cpu().numpy().astype(np.float64)


def train_residual_cnn_net(
    X_train: np.ndarray,
    Z_train: np.ndarray,
    X_val: np.ndarray,
    Z_val: np.ndarray,
    device: str = "cpu",
    epochs: int = 12,
    batch_size: int = 512,
    lr: float = 3e-4,
    hidden_channels: int = 64,
) -> nn.Module:
    net = ResidualGridCNN(in_channels=8, hidden_channels=hidden_channels).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    Xtr = torch.tensor(X_train.astype(np.float32), device=device)
    Ztr = torch.tensor(Z_train.astype(np.float32), device=device)
    Xva = torch.tensor(X_val.astype(np.float32), device=device)
    Zva = torch.tensor(Z_val.astype(np.float32), device=device)

    def make_predictions(x_batch: torch.Tensor) -> torch.Tensor:
        a, feat_grid = residual_feature_grid_torch(x_batch)
        residual_grid = net(feat_grid)
        residual = residual_grid.reshape(x_batch.shape[0], residual_mod.num_sensors)
        return a + residual

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

        net.eval()
        with torch.no_grad():
            z_hat_val = make_predictions(Xva)
            val_loss = F.mse_loss(z_hat_val, Zva).item()

        print(f"Epoch {ep:02d} | train MSE: {total_loss/max(n_batches,1):.4f} | val MSE: {val_loss:.4f}")

    return net


def main() -> None:
    seed_numpy = 0
    seed_torch = 0
    seed_sensor_params = 123
    seed_dataset_noise = 303
    seed_dataset_states = 303
    seed_tracking = 777

    np.random.seed(seed_numpy)
    torch.manual_seed(seed_torch)

    sensor_p0 = 1e4
    sensor_grid_x = 5
    sensor_grid_y = 5
    sensor_coord_min = -30.0
    sensor_coord_max = 30.0

    residual_mod.configure_sensor_grid_and_physics(
        p0=sensor_p0,
        sensors_x=sensor_grid_x,
        sensors_y=sensor_grid_y,
        coord_min=sensor_coord_min,
        coord_max=sensor_coord_max,
    )

    gain_std = 0.10
    bias_std = 0.8
    meas_std = 1.0

    shadow_sin_amplitude = 0.6
    shadow_sin_frequency = 0.12
    shadow_cos_amplitude = 0.4
    shadow_cos_frequency = 0.10

    residual_mod.configure_shadowing(
        sin_amp=shadow_sin_amplitude,
        sin_freq=shadow_sin_frequency,
        cos_amp=shadow_cos_amplitude,
        cos_freq=shadow_cos_frequency,
    )

    roi_min = -30.0
    roi_max = 30.0
    vmin = -2.0
    vmax = 2.0
    n_train = 30000
    n_val = 8000

    train_epochs = 200
    batch_size = 512
    lr = 3e-4
    hidden_channels = 64

    dt = 0.25
    tau = 1e-2
    t_steps = 80
    trajectory_kind = "cv_gaussian_extended"  # secenekler: cv_gaussian, cv_gaussian_extended
    q_scale = 1.0
    extended_process_noise_scale = 1.6
    extended_motion_bias = np.array([0.0, 0.0, -0.010, 0.006], dtype=np.float64)
    t_steps = residual_mod.default_tracking_steps_for_trajectory(trajectory_kind, base_steps=t_steps, extended_steps=110)
    x0_true = residual_mod.default_initial_state_for_trajectory(
        trajectory_kind=trajectory_kind,
        cv_x0_true=np.array([-8.0, 10.0, 1.2, -0.8], dtype=np.float64),
    )
    x0_est = residual_mod.default_filter_initial_state_for_trajectory(
        trajectory_kind=trajectory_kind,
        cv_x0_est=np.array([-6.5, 9.0, 0.6, -0.4], dtype=np.float64),
    )
    p_init = np.diag([4.0, 4.0, 1.0, 1.0]).astype(np.float64)

    ukf_alpha = 0.3
    ukf_beta = 2.0
    ukf_kappa = 0.0
    meas_std_floor = 1e-3
    r_adapt_func = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    g_i_true, b_i_true = residual_mod.make_true_sensor_params(
        seed=seed_sensor_params,
        gain_std=gain_std,
        bias_std=bias_std,
    )
    print("True continuous sensor mismatch set.")

    X_all, Z_all = residual_mod.build_dataset(
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

    print("Training residual measurement CNN...")
    net = train_residual_cnn_net(
        X_train,
        Z_train,
        X_val,
        Z_val,
        device=device,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_channels=hidden_channels,
    )

    Q = q_scale * residual_mod.Q_white_accel(dt, tau)
    print("Q scale:", q_scale)
    trajectory_scenario = residual_mod.build_configured_trajectory_scenario(
        trajectory_kind=trajectory_kind,
        dt=dt,
        cv_process_noise_cov=Q,
        extended_process_noise_scale=extended_process_noise_scale,
        extended_motion_bias=extended_motion_bias,
    )
    print("Trajectory scenario:", trajectory_kind)

    xs_true, zs_meas = residual_mod.simulate_trajectory_and_measurements(
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

    def hx_phys(x_np: np.ndarray) -> np.ndarray:
        return residual_mod.hx_meas_physics_only(x_np)

    def hx_hybrid_cnn(x_np: np.ndarray) -> np.ndarray:
        return hx_meas_hybrid_cnn(x_np, net=net, device=device)

    print("Running UKF (physics-only)...")
    xs_est_phys = residual_mod.run_ukf_tracking(
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

    print("Running UKF (hybrid physics + CNN residual)...")
    xs_est_hyb = residual_mod.run_ukf_tracking(
        xs_true,
        zs_meas,
        dt,
        Q,
        x0_est,
        p_init,
        hx_hybrid_cnn,
        meas_std=meas_std,
        meas_std_floor=meas_std_floor,
        ukf_alpha=ukf_alpha,
        ukf_beta=ukf_beta,
        ukf_kappa=ukf_kappa,
        R_adapt_func=r_adapt_func,
    )

    rmse_phys = residual_mod.rmse_pos(xs_est_phys, xs_true)
    rmse_hyb = residual_mod.rmse_pos(xs_est_hyb, xs_true)
    print(f"Position RMSE | physics-only UKF: {rmse_phys:.4f}")
    print(f"Position RMSE | hybrid UKF      : {rmse_hyb:.4f}")

    plt.figure(figsize=(10, 8))
    plt.plot(xs_true[:, 0], xs_true[:, 1], "r-", linewidth=2.0, label="True")
    plt.plot(xs_est_phys[:, 0], xs_est_phys[:, 1], "k--", linewidth=2.0, label="UKF (physics-only)")
    plt.plot(xs_est_hyb[:, 0], xs_est_hyb[:, 1], color="tab:cyan", linewidth=2.0, label="UKF (hybrid physics+CNN)")
    plt.scatter(
        residual_mod.sensor_coords[:, 0],
        residual_mod.sensor_coords[:, 1],
        s=80,
        marker="^",
        alpha=0.7,
        label="Sensors",
    )
    plt.title("Continuous Sensor Tracking: UKF with Physics vs Hybrid Physics+CNN")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    err_phys = np.linalg.norm(xs_est_phys[:, :2] - xs_true[:, :2], axis=1)
    err_hyb = np.linalg.norm(xs_est_hyb[:, :2] - xs_true[:, :2], axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(err_phys, "k--", label="pos error (physics-only)")
    plt.plot(err_hyb, color="tab:cyan", label="pos error (hybrid CNN)")
    plt.title("Position Error vs Time")
    plt.xlabel("t")
    plt.ylabel("||e_pos||")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
