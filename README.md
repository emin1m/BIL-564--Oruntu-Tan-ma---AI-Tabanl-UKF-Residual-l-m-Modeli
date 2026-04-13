AI-Assisted Measurement Models in UKF

Residual vs Direct Neural Measurement for State Estimation

Overview

This project improves state estimation by enhancing the measurement function in an Unscented Kalman Filter (UKF).

Instead of modifying the filter itself, we improve:

the measurement model

Motivation

Real-world sensors are not ideal:

Gain variations
Bias offsets
Environmental effects (shadowing, multipath)
Measurement noise

A purely physics-based model cannot capture all these effects.

Approach

We compare three measurement strategies inside the same UKF.

1. Physics-Only Model
z = h_physics(x)
Pure analytical model
Baseline approach
2. Hybrid Residual Model (Physics + AI)
z = h_physics(x) + g_theta(phi(x))
Neural network learns the residual error
Keeps physical structure
More stable
3. Direct Neural Measurement Model
z = h_theta(x)
Fully learned measurement function
No physics assumptions
More flexible but harder to train
Key Idea

We do NOT change the Kalman filter
We improve the measurement function

System Model

State:

[x, y, vx, vy]

Physics Model:

a_i(x) = sqrt(P0 / (1 + d_i^2))

True Measurement Model:

z_i = g_i * a_i(x) + b_i + shadow(x) + noise

Where:

g_i → sensor gain
b_i → sensor bias
shadow(x) → position-dependent distortion
noise → Gaussian noise
Experiments
Same dataset for all models
Same trajectory and noise
Fair comparison

Metrics:

Position RMSE
Tracking accuracy over time
Results
Physics-only model fails under mismatch
Residual model significantly improves performance
Direct model can outperform but is less stable
Project Structure
.
├── hybrid_ukf_residual_tracking.py
├── hybrid_ukf_direct_h_tracking.py
├── hybrid_ukf_residual_cnn_tracking.py
├── hybrid_ukf_residual_gru_tracking.py
├── hybrid_ukf_compare_residual_vs_direct.py
├── hybrid_ukf_monte_carlo_time_mse.py
├── trajectory_scenarios.py
├── trajectory_scenarios_gaussian_extended.py
How to Run

Train and compare:

python hybrid_ukf_compare_residual_vs_direct.py

Monte Carlo:

python hybrid_ukf_monte_carlo_time_mse.py
Why UKF (not EKF)?
Neural network inside measurement
No analytical Jacobian
UKF works without derivatives
