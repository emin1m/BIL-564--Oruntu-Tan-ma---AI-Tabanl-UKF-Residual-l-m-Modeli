AI-Assisted Measurement Models in UKF

Improving state estimation by learning better measurement functions inside a UKF.

Idea

We do NOT change the filter.
We improve the measurement model.

Models

1. Physics-only

z = h_physics(x)

2. Hybrid (Residual)

z = h_physics(x) + g_theta(phi(x))

3. Direct (Neural)

z = h_theta(x)
System
State: [x, y, vx, vy]

Physics:
a_i(x) = sqrt(P0 / (1 + d_i^2))

True measurement:
z_i = g_i * a_i(x) + b_i + shadow(x) + noise  
Result  
Physics-only → fails under mismatch  
Hybrid → best balance  
Direct → powerful but less stable  
