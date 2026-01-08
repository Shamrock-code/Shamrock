"""
SR-GSPH Tests (Kitajima et al. 2025 - arXiv:2510.18251v1)
=========================================================

Special Relativistic Godunov SPH test suite based on the Kitajima paper.

1D Riemann Problems:
  - Problem 1: Sod Shock Tube (problem1_sod.py)
      (P_L, n_L) = (1.0, 1.0), (P_R, n_R) = (0.1, 0.125)
      γ = 5/3, t = 0.35

  - Problem 2: Standard Relativistic Blast Wave (problem2_blast.py)
      (P_L, n_L) = (40/3, 10), (P_R, n_R) = (1e-6, 1)
      γ = 5/3, t = 0.4

  - Problem 3: Strong Relativistic Blast Wave (problem3_strong_blast.py)
      (P_L, n_L) = (1000, 1), (P_R, n_R) = (0.01, 1)
      γ = 5/3, t = 0.16

  - Problem 4: Ultra-Relativistic Shock (problem4_ultra_relativistic.py)
      Left moving at v = 0.9 to 0.999999999
      γ = 5/3, t = 0.3

  - Problem 5: Tangential Velocity Tests (problem5_tangent_velocity.py)
      (P_L, n_L) = (1000, 1), (P_R, n_R) = (0.01, 1)
      vt = 0, 0.9, 0.99

2D Problems:
  - Problem 6: 2D Sod Problem (problem6_2d_sod.py)
      Same as Problem 1, periodic y-boundary

  - Problem 7: 2D Kelvin-Helmholtz Instability (problem7_kh_instability.py)
      P = 1, n = 0.5, v_shear = ±0.3
      Sinusoidal perturbation A0 = 1/40, λ = 1/3

Utilities:
  - kitajima_plotting.py: Exact solutions and Kitajima-style plotting
"""
