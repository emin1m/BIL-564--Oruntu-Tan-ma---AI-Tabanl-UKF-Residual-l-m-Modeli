from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from trajectory_scenarios import (
    build_process_noise_from_accel_distribution as build_process_noise_from_accel_distribution_shared,
    build_configured_trajectory_scenario,
    default_filter_initial_state_for_trajectory,
    default_initial_state_for_trajectory,
    default_tracking_steps_for_trajectory,
)

try:
    import torch
except ImportError as e:
    raise ImportError("This script requires PyTorch. Install it, then rerun.") from e

import hybrid_ukf_residual_tracking as residual_mod
import hybrid_ukf_residual_gru_tracking as residual_gru_mod
import hybrid_ukf_residual_cnn_tracking as residual_cnn_mod

import hybrid_ukf_direct_h_tracking as direct_mod
def build_process_noise_from_accel_distribution(
    dt: float,
    accel_std_x: float,
    accel_std_y: float,
    accel_corr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Physical process noise model:
      a_k ~ N(0, Sigma_a)
      x_{k+1} = F x_k + G a_k

    Returns both the acceleration covariance Sigma_a and the equivalent state
    covariance Q = G Sigma_a G^T used by the UKF.
    """
    return build_process_noise_from_accel_distribution_shared(
        dt=dt,
        accel_std_x=accel_std_x,
        accel_std_y=accel_std_y,
        accel_corr=accel_corr,
    )
def set_shared_sensor_grid() -> None:
    """
    Force both modules to use exactly the same sensor geometry.
    """
    num_sensors_x = 5
    num_sensors_y = 5
    rng_values = np.linspace(-30, 30, num_sensors_x)
    sensor_coords = np.array([[x, y] for x in rng_values for y in rng_values], dtype=np.float64)

    for mod in (residual_mod, direct_mod):
        mod.num_sensors_x = num_sensors_x
        mod.num_sensors_y = num_sensors_y
        mod.num_sensors = num_sensors_x * num_sensors_y
        mod.rng_values = rng_values.copy()
        mod.sensor_coords = sensor_coords.copy()


def build_shared_dataset(
    N: int,
    g_i: np.ndarray,
    b_i: np.ndarray,
    meas_std: float,
    seed: int,
    roi_min: float = -30.0,
    roi_max: float = 30.0,
    vmin: float = -2.0,
    vmax: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shared dataset for both methods so comparison is fair.
    """
    rng = np.random.default_rng(seed)
    X = residual_mod.sample_random_states(
        N=N,
        roi_min=roi_min,
        roi_max=roi_max,
        vmin=vmin,
        vmax=vmax,
        seed=seed,
    )

    Z = np.zeros((N, residual_mod.num_sensors), dtype=np.float64)
    for i in range(N):
        Z[i] = residual_mod.sample_measurement_np(
            X[i],
            g_i,
            b_i,
            meas_std=meas_std,
            seed=int(rng.integers(1, 1_000_000_000)),
        )
    return X, Z


def main() -> None:
    # -----------------------------
    # Shared setup (same params for both methods)
    # -----------------------------
    set_shared_sensor_grid()

    np.random.seed(0)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Keep all key parameters aligned across both methods.
    gain_std = 0.10
    bias_std = 0.8
    meas_std = 0.5
    N_train = 30000
    N_val = 8000
    train_epochs = 50
    batch_size = 512
    lr = 3e-4

    dt = 0.25
    accel_std_x = 0.20
    accel_std_y = 0.20
    accel_corr = 0.15
    T_steps = 80
    trajectory_kind = "cv_gaussian_extended"  # secenekler: cv_gaussian, cv_gaussian_extended
    q_scale = 1.0
    extended_process_noise_scale = 1.6
    extended_motion_bias = np.array([0.0, 0.0, -0.010, 0.006], dtype=np.float64)
    T_steps = default_tracking_steps_for_trajectory(trajectory_kind, base_steps=T_steps, extended_steps=80)
    x0_true = default_initial_state_for_trajectory(
        trajectory_kind=trajectory_kind,
        cv_x0_true=np.array([-8.0, 10.0, 1.2, -0.8], dtype=np.float64),
    )
    x0_est = default_filter_initial_state_for_trajectory(
        trajectory_kind=trajectory_kind,
        cv_x0_est=np.array([-6.5, 9.0, 0.6, -0.4], dtype=np.float64),
    )
    P_init = np.diag([4.0, 4.0, 1.0, 1.0]).astype(np.float64)

    seed_params = 123
    seed_dataset = 303
    seed_traj = 777

    # -----------------------------
    # True mismatch model params
    # -----------------------------
    g_i_true, b_i_true = residual_mod.make_true_sensor_params(
        seed=seed_params,
        gain_std=gain_std,
        bias_std=bias_std,
    )
    print("True continuous sensor mismatch set.")

    # -----------------------------
    # Shared training data
    # -----------------------------
    X_all, Z_all = build_shared_dataset(
        N=N_train + N_val,
        g_i=g_i_true,
        b_i=b_i_true,
        meas_std=meas_std,
        seed=seed_dataset,
        roi_min=-30.0,
        roi_max=30.0,
        vmin=-2.0,
        vmax=2.0,
    )
    X_train, Z_train = X_all[:N_train], Z_all[:N_train]
    X_val, Z_val = X_all[N_train:], Z_all[N_train:]

    # -----------------------------
    # Train residual model
    # -----------------------------
    print("\nTraining residual model...")
    net_residual = residual_mod.train_residual_net(
        X_train,
        Z_train,
        X_val,
        Z_val,
        device=device,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # -----------------------------
    # Train residual GRU model
    # -----------------------------
    print("\nTraining residual GRU model...")
    net_residual_gru = residual_gru_mod.train_residual_gru_net(
        X_train,
        Z_train,
        X_val,
        Z_val,
        device=device,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # -----------------------------
    # Train residual CNN model
    # -----------------------------
    print("\nTraining residual CNN model...")
    net_residual_cnn = residual_cnn_mod.train_residual_cnn_net(
        X_train,
        Z_train,
        X_val,
        Z_val,
        device=device,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
    )


    # -----------------------------
    # Train direct-h model
    # -----------------------------
    print("\nTraining direct-h model...")
    net_direct = direct_mod.train_direct_h_net(
        X_train,
        Z_train,
        X_val,
        Z_val,
        device=device,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # -----------------------------
    # Shared tracking data
    # -----------------------------
    accel_cov, Q = build_process_noise_from_accel_distribution(
        dt=dt,
        accel_std_x=accel_std_x,
        accel_std_y=accel_std_y,
        accel_corr=accel_corr,
    )
    Q = q_scale * Q
    print("Q scale:", q_scale)
    print(
        "\nPhysical process noise: "
        f"a_k ~ N(0, Sigma_a), std=({accel_std_x:.3f}, {accel_std_y:.3f}), corr={accel_corr:.2f}"
    )
    trajectory_scenario = build_configured_trajectory_scenario(
        trajectory_kind=trajectory_kind,
        dt=dt,
        cv_process_noise_cov=Q,
        extended_process_noise_scale=extended_process_noise_scale,
        extended_motion_bias=extended_motion_bias,
    )
    print("Trajectory scenario:", trajectory_kind)
    xs_true, zs_meas = residual_mod.simulate_trajectory_and_measurements(
        T=T_steps,
        dt=dt,
        x0_true=x0_true,
        Q=Q,
        g_i=g_i_true,
        b_i=b_i_true,
        meas_std=meas_std,
        seed=seed_traj,
        trajectory_scenario=trajectory_scenario,
    )

    def hx_phys(x_np: np.ndarray) -> np.ndarray:
        return residual_mod.hx_meas_physics_only(x_np)

    def hx_residual(x_np: np.ndarray) -> np.ndarray:
        return residual_mod.hx_meas_hybrid(x_np, net=net_residual, device=device)

    def hx_residual_gru(x_np: np.ndarray) -> np.ndarray:
        return residual_gru_mod.hx_meas_hybrid_gru(x_np, net=net_residual_gru, device=device)

    def hx_residual_cnn(x_np: np.ndarray) -> np.ndarray:
        return residual_cnn_mod.hx_meas_hybrid_cnn(x_np, net=net_residual_cnn, device=device)

    def hx_direct(x_np: np.ndarray) -> np.ndarray:
        return direct_mod.hx_meas_direct(x_np, net=net_direct, device=device)

    # Use same UKF runner and same Q/R settings for all six runs.
    print("\nRunning UKF (physics-only)...")
    xs_est_phys = residual_mod.run_ukf_tracking(
        xs_true,
        zs_meas,
        dt,
        Q,
        x0_est,
        P_init,
        hx_phys,
        meas_std=meas_std,
    )

    print("Running UKF (residual-hybrid)...")
    xs_est_res = residual_mod.run_ukf_tracking(
        xs_true,
        zs_meas,
        dt,
        Q,
        x0_est,
        P_init,
        hx_residual,
        meas_std=meas_std,
    )

    print("Running UKF (residual-GRU)...")
    xs_est_res_gru = residual_mod.run_ukf_tracking(
        xs_true,
        zs_meas,
        dt,
        Q,
        x0_est,
        P_init,
        hx_residual_gru,
        meas_std=meas_std,
    )

    print("Running UKF (residual-CNN)...")
    xs_est_res_cnn = residual_mod.run_ukf_tracking(
        xs_true,
        zs_meas,
        dt,
        Q,
        x0_est,
        P_init,
        hx_residual_cnn,
        meas_std=meas_std,
    )



    print("Running UKF (direct-h)...")
    xs_est_dir = residual_mod.run_ukf_tracking(
        xs_true,
        zs_meas,
        dt,
        Q,
        x0_est,
        P_init,
        hx_direct,
        meas_std=meas_std,
    )

    rmse_phys = residual_mod.rmse_pos(xs_est_phys, xs_true)
    rmse_res = residual_mod.rmse_pos(xs_est_res, xs_true)
    rmse_res_gru = residual_mod.rmse_pos(xs_est_res_gru, xs_true)
    rmse_res_cnn = residual_mod.rmse_pos(xs_est_res_cnn, xs_true)
    rmse_dir = residual_mod.rmse_pos(xs_est_dir, xs_true)

    print("\n=== Comparison (same parameters, same data) ===")
    print(f"RMSE | physics-only  : {rmse_phys:.4f}")
    print(f"RMSE | residual-hybrid: {rmse_res:.4f}")
    print(f"RMSE | residual-gru   : {rmse_res_gru:.4f}")
    print(f"RMSE | residual-cnn   : {rmse_res_cnn:.4f}")
    print(f"RMSE | direct-h       : {rmse_dir:.4f}")

    # -----------------------------
    # Plots
    # -----------------------------
    plt.figure(figsize=(10, 8))
    plt.plot(xs_true[:, 0], xs_true[:, 1], "r-", linewidth=2.0, label="True")
    plt.plot(xs_est_phys[:, 0], xs_est_phys[:, 1], "k--", linewidth=2.0, label="UKF (physics-only)")
    plt.plot(xs_est_res[:, 0], xs_est_res[:, 1], "g-", linewidth=2.0, label="UKF (residual-hybrid)")
    plt.plot(
        xs_est_res_gru[:, 0],
        xs_est_res_gru[:, 1],
        color="tab:orange",
        linewidth=2.0,
        label="UKF (residual-GRU)",
    )
    plt.plot(
        xs_est_res_cnn[:, 0],
        xs_est_res_cnn[:, 1],
        color="tab:cyan",
        linewidth=2.0,
        label="UKF (residual-CNN)",
    )

    plt.plot(xs_est_dir[:, 0], xs_est_dir[:, 1], "b-", linewidth=2.0, label="UKF (direct-h)")
    plt.scatter(residual_mod.sensor_coords[:, 0], residual_mod.sensor_coords[:, 1], s=80, marker="^", alpha=0.7, label="Sensors")
    plt.title("Hybrid UKF Comparison: Residual-MLP vs GRU vs CNN vs CNN+GRU vs Direct-h")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    err_phys = np.linalg.norm(xs_est_phys[:, :2] - xs_true[:, :2], axis=1)
    err_res = np.linalg.norm(xs_est_res[:, :2] - xs_true[:, :2], axis=1)
    err_res_gru = np.linalg.norm(xs_est_res_gru[:, :2] - xs_true[:, :2], axis=1)
    err_res_cnn = np.linalg.norm(xs_est_res_cnn[:, :2] - xs_true[:, :2], axis=1)

    err_dir = np.linalg.norm(xs_est_dir[:, :2] - xs_true[:, :2], axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(err_phys, "k--", label="pos error (physics-only)")
    plt.plot(err_res, "g-", label="pos error (residual-hybrid)")
    plt.plot(err_res_gru, color="tab:orange", label="pos error (residual-GRU)")
    plt.plot(err_res_cnn, color="tab:cyan", label="pos error (residual-CNN)")
    plt.plot(err_dir, "b-", label="pos error (direct-h)")
    plt.title("Position Error vs Time")
    plt.xlabel("t")
    plt.ylabel("||e_pos||")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
