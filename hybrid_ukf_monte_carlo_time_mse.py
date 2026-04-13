from __future__ import annotations

from pathlib import Path
from typing import Callable

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
    import torch.nn as nn
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
    return build_process_noise_from_accel_distribution_shared(
        dt=dt,
        accel_std_x=accel_std_x,
        accel_std_y=accel_std_y,
        accel_corr=accel_corr,
    )
def set_shared_sensor_grid(
    sensors_x: int,
    sensors_y: int,
    coord_min: float,
    coord_max: float,
    p0: float,
) -> None:
    residual_mod.configure_sensor_grid_and_physics(
        p0=p0,
        sensors_x=sensors_x,
        sensors_y=sensors_y,
        coord_min=coord_min,
        coord_max=coord_max,
    )

    rng_values = np.linspace(coord_min, coord_max, sensors_x)
    sensor_coords = np.array([[x, y] for x in rng_values for y in rng_values], dtype=np.float64)

    direct_mod.P0 = float(p0)
    direct_mod.num_sensors_x = int(sensors_x)
    direct_mod.num_sensors_y = int(sensors_y)
    direct_mod.num_sensors = sensors_x * sensors_y
    direct_mod.rng_values = rng_values.copy()
    direct_mod.sensor_coords = sensor_coords.copy()


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


def instantiate_model(
    model_kind: str,
    device: str,
    residual_hidden: int,
    residual_gru_hidden: int,
    residual_cnn_channels: int,
    residual_cnn_gru_cnn_channels: int,
    residual_cnn_gru_hidden: int,
) -> nn.Module:
    if model_kind == "residual_mlp":
        return residual_mod.ResidualPerSensorMLP(in_dim=8, hidden=residual_hidden).to(device)
    if model_kind == "residual_gru":
        return residual_gru_mod.ResidualPerSensorGRU(in_dim=8, hidden=residual_gru_hidden).to(device)
    if model_kind == "residual_cnn":
        return residual_cnn_mod.ResidualGridCNN(in_channels=8, hidden_channels=residual_cnn_channels).to(device)

    if model_kind == "direct_h":
        return direct_mod.DirectHPerSensorMLP(in_dim=7, hidden=128).to(device)
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def train_model(
    model_kind: str,
    X_train: np.ndarray,
    Z_train: np.ndarray,
    X_val: np.ndarray,
    Z_val: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    residual_hidden: int,
    residual_gru_hidden: int,
    residual_cnn_channels: int,
    residual_cnn_gru_cnn_channels: int,
    residual_cnn_gru_hidden: int,
) -> nn.Module:
    if model_kind == "residual_mlp":
        return residual_mod.train_residual_net(
            X_train,
            Z_train,
            X_val,
            Z_val,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_hidden=residual_hidden,
        )
    if model_kind == "residual_gru":
        return residual_gru_mod.train_residual_gru_net(
            X_train,
            Z_train,
            X_val,
            Z_val,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_hidden=residual_gru_hidden,
        )
    if model_kind == "residual_cnn":
        return residual_cnn_mod.train_residual_cnn_net(
            X_train,
            Z_train,
            X_val,
            Z_val,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_channels=residual_cnn_channels,
        )

    if model_kind == "direct_h":
        return direct_mod.train_direct_h_net(
            X_train,
            Z_train,
            X_val,
            Z_val,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def make_hx_callable(model_kind: str, net: nn.Module, device: str) -> Callable[[np.ndarray], np.ndarray]:
    if model_kind == "residual_mlp":
        return lambda x_np: residual_mod.hx_meas_hybrid(x_np, net=net, device=device)
    if model_kind == "residual_gru":
        return lambda x_np: residual_gru_mod.hx_meas_hybrid_gru(x_np, net=net, device=device)
    if model_kind == "residual_cnn":
        return lambda x_np: residual_cnn_mod.hx_meas_hybrid_cnn(x_np, net=net, device=device)

    if model_kind == "direct_h":
        return lambda x_np: direct_mod.hx_meas_direct(x_np, net=net, device=device)
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def compute_position_se_over_time(xs_est: np.ndarray, xs_true: np.ndarray) -> np.ndarray:
    diff_xy = xs_est[:, :2] - xs_true[:, :2]
    return np.sum(diff_xy**2, axis=1)


def main() -> None:
    # -----------------------------
    # Experiment selection
    # -----------------------------
    model_kinds = [
        "residual_mlp",
        "residual_gru",
        "residual_cnn",
        "direct_h",
    ]
    num_monte_carlo_runs = 100

    # Monte Carlo itself does not require saving weights because the model is
    # trained once and then reused in memory. Saving is optional for later reuse.
    save_model_weights = True
    load_saved_weights_if_available = False

    # -----------------------------
    # Reproducibility
    # -----------------------------
    seed_numpy = 0
    seed_torch = 0
    seed_sensor_params = 123
    seed_dataset = 303
    seed_mc_base = 10_000

    np.random.seed(seed_numpy)
    torch.manual_seed(seed_torch)

    # -----------------------------
    # Sensor + mismatch config
    # -----------------------------
    sensor_p0 = 1e4
    sensor_grid_x = 5
    sensor_grid_y = 5
    sensor_coord_min = -30.0
    sensor_coord_max = 30.0

    gain_std = 0.10
    bias_std = 0.8
    meas_std = 0.5

    shadow_sin_amplitude = 0.6
    shadow_sin_frequency = 0.12
    shadow_cos_amplitude = 0.4
    shadow_cos_frequency = 0.10

    # -----------------------------
    # Train/val dataset config
    # -----------------------------
    roi_min = -30.0
    roi_max = 30.0
    vmin = -2.0
    vmax = 2.0
    n_train = 30000
    n_val = 8000

    # -----------------------------
    # Model hyper-parameters
    # -----------------------------
    train_epochs = 50
    batch_size = 512
    lr = 3e-4

    residual_hidden = 128
    residual_gru_hidden = 128
    residual_cnn_channels = 64
    residual_cnn_gru_cnn_channels = 32
    residual_cnn_gru_hidden = 64

    # -----------------------------
    # Tracking config
    # -----------------------------
    dt = 0.25
    accel_std_x = 0.20
    accel_std_y = 0.20
    accel_corr = 0.15
    t_steps = 60
    trajectory_kind = "cv_gaussian_extended"  # secenekler: cv_gaussian, cv_gaussian_extended
    q_scale = 1.0
    extended_process_noise_scale = 1.6
    extended_motion_bias = np.array([0.0, 0.0, -0.010, 0.006], dtype=np.float64)
    t_steps = default_tracking_steps_for_trajectory(trajectory_kind, base_steps=t_steps, extended_steps=60)
    x0_true = default_initial_state_for_trajectory(
        trajectory_kind=trajectory_kind,
        cv_x0_true=np.array([-8.0, 10.0, 1.2, -0.8], dtype=np.float64),
    )
    x0_est = default_filter_initial_state_for_trajectory(
        trajectory_kind=trajectory_kind,
        cv_x0_est=np.array([-6.5, 9.0, 0.6, -0.4], dtype=np.float64),
    )
    p_init = np.diag([4.0, 4.0, 1.0, 1.0]).astype(np.float64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Selected models:", ", ".join(model_kinds))

    set_shared_sensor_grid(
        sensors_x=sensor_grid_x,
        sensors_y=sensor_grid_y,
        coord_min=sensor_coord_min,
        coord_max=sensor_coord_max,
        p0=sensor_p0,
    )

    residual_mod.configure_shadowing(
        sin_amp=shadow_sin_amplitude,
        sin_freq=shadow_sin_frequency,
        cos_amp=shadow_cos_amplitude,
        cos_freq=shadow_cos_frequency,
    )

    g_i_true, b_i_true = residual_mod.make_true_sensor_params(
        seed=seed_sensor_params,
        gain_std=gain_std,
        bias_std=bias_std,
    )

    X_all, Z_all = build_shared_dataset(
        N=n_train + n_val,
        g_i=g_i_true,
        b_i=b_i_true,
        meas_std=meas_std,
        seed=seed_dataset,
        roi_min=roi_min,
        roi_max=roi_max,
        vmin=vmin,
        vmax=vmax,
    )
    X_train, Z_train = X_all[:n_train], Z_all[:n_train]
    X_val, Z_val = X_all[n_train:], Z_all[n_train:]

    weights_dir = Path(__file__).resolve().parent / "trained_weights"
    trained_models: dict[str, nn.Module] = {}
    hx_by_model: dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    for model_kind in model_kinds:
        weights_path = weights_dir / f"{model_kind}.pt"

        if load_saved_weights_if_available and weights_path.exists():
            net = instantiate_model(
                model_kind=model_kind,
                device=device,
                residual_hidden=residual_hidden,
                residual_gru_hidden=residual_gru_hidden,
                residual_cnn_channels=residual_cnn_channels,
                residual_cnn_gru_cnn_channels=residual_cnn_gru_cnn_channels,
                residual_cnn_gru_hidden=residual_cnn_gru_hidden,
            )
            checkpoint = torch.load(weights_path, map_location=device)
            net.load_state_dict(checkpoint["model_state_dict"])
            net.eval()
            print(f"Loaded weights for {model_kind}: {weights_path}")
        else:
            print(f"Training model: {model_kind}")
            net = train_model(
                model_kind=model_kind,
                X_train=X_train,
                Z_train=Z_train,
                X_val=X_val,
                Z_val=Z_val,
                device=device,
                epochs=train_epochs,
                batch_size=batch_size,
                lr=lr,
                residual_hidden=residual_hidden,
                residual_gru_hidden=residual_gru_hidden,
                residual_cnn_channels=residual_cnn_channels,
                residual_cnn_gru_cnn_channels=residual_cnn_gru_cnn_channels,
                residual_cnn_gru_hidden=residual_cnn_gru_hidden,
            )
            if save_model_weights:
                weights_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_kind": model_kind,
                        "model_state_dict": net.state_dict(),
                    },
                    weights_path,
                )
                print(f"Saved weights for {model_kind}: {weights_path}")

        net.eval()
        trained_models[model_kind] = net
        hx_by_model[model_kind] = make_hx_callable(model_kind, net, device)

    accel_cov, Q = build_process_noise_from_accel_distribution(
        dt=dt,
        accel_std_x=accel_std_x,
        accel_std_y=accel_std_y,
        accel_corr=accel_corr,
    )
    Q = q_scale * Q
    print("Q scale:", q_scale)
    trajectory_scenario = build_configured_trajectory_scenario(
        trajectory_kind=trajectory_kind,
        dt=dt,
        cv_process_noise_cov=Q,
        extended_process_noise_scale=extended_process_noise_scale,
        extended_motion_bias=extended_motion_bias,
    )
    print("Trajectory scenario:", trajectory_kind)

    se_all_by_model = {
        model_kind: np.zeros((num_monte_carlo_runs, t_steps), dtype=np.float64)
        for model_kind in model_kinds
    }

    for mc_idx in range(num_monte_carlo_runs):
        run_seed = seed_mc_base + mc_idx
        xs_true, zs_meas = residual_mod.simulate_trajectory_and_measurements(
            T=t_steps,
            dt=dt,
            x0_true=x0_true,
            Q=Q,
            g_i=g_i_true,
            b_i=b_i_true,
            meas_std=meas_std,
            seed=run_seed,
            trajectory_scenario=trajectory_scenario,
        )

        for model_kind in model_kinds:
            xs_est_model = residual_mod.run_ukf_tracking(
                xs_true,
                zs_meas,
                dt,
                Q,
                x0_est,
                p_init,
                hx_by_model[model_kind],
                meas_std=meas_std,
            )
            se_all_by_model[model_kind][mc_idx] = compute_position_se_over_time(xs_est_model, xs_true)

        if (mc_idx + 1) % 10 == 0 or mc_idx == 0:
            print(f"Monte Carlo run {mc_idx + 1}/{num_monte_carlo_runs}")

    mse_by_model = {model_kind: se_all_by_model[model_kind].mean(axis=0) for model_kind in model_kinds}
    rmse_by_model = {model_kind: np.sqrt(mse_by_model[model_kind]) for model_kind in model_kinds}

    print("\n=== Monte Carlo Summary ===")
    print(f"Monte Carlo runs          : {num_monte_carlo_runs}")
    for model_kind in model_kinds:
        mse_model_t = mse_by_model[model_kind]
        rmse_model_t = rmse_by_model[model_kind]
        print(
            f"{model_kind:20s} | "
            f"avg MSE: {mse_model_t.mean():.6f} | "
            f"avg RMSE: {rmse_model_t.mean():.6f} | "
            f"final MSE: {mse_model_t[-1]:.6f} | "
            f"final RMSE: {rmse_model_t[-1]:.6f}"
        )

    results_dir = Path(__file__).resolve().parent / "mc_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"all_models_mc_{num_monte_carlo_runs}.npz"
    arrays_to_save: dict[str, np.ndarray] = {
        "model_kinds": np.array(model_kinds, dtype="<U32"),
        "num_monte_carlo_runs": np.array([num_monte_carlo_runs], dtype=np.int64),
    }
    for model_kind in model_kinds:
        arrays_to_save[f"mse_t_{model_kind}"] = mse_by_model[model_kind]
        arrays_to_save[f"rmse_t_{model_kind}"] = rmse_by_model[model_kind]
        arrays_to_save[f"se_all_{model_kind}"] = se_all_by_model[model_kind]
    np.savez(results_path, **arrays_to_save)
    print(f"Saved Monte Carlo arrays to: {results_path}")

    t_axis = np.arange(t_steps)

    plt.figure(figsize=(10, 4))
    for model_kind in model_kinds:
        plt.plot(t_axis, mse_by_model[model_kind], linewidth=2.0, label=f"MSE(t) | {model_kind}")
    plt.title(f"Monte Carlo Position MSE per Time Step ({num_monte_carlo_runs} runs)")
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for model_kind in model_kinds:
        plt.plot(t_axis, rmse_by_model[model_kind], linewidth=2.0, label=f"RMSE(t) | {model_kind}")
    plt.title(f"Monte Carlo Position RMSE per Time Step ({num_monte_carlo_runs} runs)")
    plt.xlabel("t")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
