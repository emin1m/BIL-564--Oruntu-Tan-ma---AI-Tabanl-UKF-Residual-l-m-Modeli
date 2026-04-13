AI-Assisted Measurement Models in UKF

Residual vs Direct Neural Measurement for State Estimation

Overview

This project explores how to improve state estimation by enhancing the measurement model in an Unscented Kalman Filter (UKF).

Instead of modifying the filter itself, we focus on a more critical component:

The measurement function 
ℎ
(
𝑥
)
h(x)

We introduce neural network–based measurement models and compare them against the classical physics-based approach.

Motivation

In real-world systems, measurement models are often imperfect due to:

Sensor gain variations
Bias offsets
Environmental effects (e.g., shadowing, multipath)
Measurement noise

A purely physics-based model cannot capture all these effects.

This project investigates:

Can we learn the mismatch between the physics model and reality using AI?

Approach

We compare three different measurement modeling strategies inside the same UKF framework:

1. Physics-Only Model
𝑧
=
ℎ
physics
(
𝑥
)
z=h
physics
	​

(x)
Uses only the analytical model
Serves as the baseline
2. Hybrid Residual Model (Physics + AI)
𝑧
=
ℎ
physics
(
𝑥
)
+
𝑔
𝜃
(
𝜙
(
𝑥
)
)
z=h
physics
	​

(x)+g
θ
	​

(ϕ(x))
Neural network learns the residual error
Combines interpretability + flexibility
More stable in practice
3. Direct Neural Measurement Model
𝑧
=
ℎ
𝜃
(
𝑥
)
z=h
θ
	​

(x)
Neural network learns the full measurement function
No physics assumptions
More expressive but harder to train
Key Idea

We do NOT change the Kalman filter.
We improve the measurement model.

This allows us to keep:

UKF structure
Probabilistic framework
Physical interpretability

while gaining:

Adaptability to real-world imperfections
Data-driven corrections
System Model
State: 
[
𝑥
,
𝑦
,
𝑣
𝑥
,
𝑣
𝑦
]
[x,y,v
x
	​

,v
y
	​

]
Motion Model: Constant velocity with process noise
Sensors: 2D grid (e.g., 5×5)
Physics Model:
𝑎
𝑖
(
𝑥
)
=
𝑃
0
1
+
𝑑
𝑖
2
a
i
	​

(x)=
1+d
i
2
	​

P
0
	​

	​

	​

True Measurement Model:
𝑧
𝑖
=
𝑔
𝑖
⋅
𝑎
𝑖
(
𝑥
)
+
𝑏
𝑖
+
shadow
(
𝑥
)
+
𝜖
z
i
	​

=g
i
	​

⋅a
i
	​

(x)+b
i
	​

+shadow(x)+ϵ
Experiments
Shared dataset for fair comparison
Same:
sensor configuration
trajectory
noise conditions
Models evaluated using:
position RMSE
tracking accuracy over time
Results

Key observations:

Physics-only model fails under mismatch
Residual model significantly improves performance
Direct model can outperform but is less stable
