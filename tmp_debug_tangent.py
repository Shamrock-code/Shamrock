"""
Quick debug script to analyze tangent velocity behavior in SR Riemann solver.
"""

import sys

sys.path.insert(0, "/Users/guo/Downloads/sph-simulators/docs/papers/sg-gsph/srrp")
import matplotlib.pyplot as plt
import numpy as np
from srrp.Solver import Solver
from srrp.State import State

# Problem 5 parameters
gamma = 5.0 / 3.0
n_L, n_R = 1.0, 1.0
P_L, P_R = 1000.0, 0.01
t_target = 0.16

# Test different tangent velocities
v_t_cases = [0.0, 0.9, 0.99]

fig, axes = plt.subplots(len(v_t_cases), 4, figsize=(16, 12))
x_exact = np.linspace(-0.5, 0.5, 1000)
xi = x_exact / t_target

solver = Solver()

for i, v_t in enumerate(v_t_cases):
    # Solve exact Riemann problem
    stateL = State(rho=n_L, vx=0.0, vt=v_t, pressure=P_L)
    stateR = State(rho=n_R, vx=0.0, vt=v_t, pressure=P_R)
    wavefan = solver.solve(stateL, stateR, gamma)

    state_exact = wavefan.getState(xi)
    P_exact = state_exact.pressure
    n_exact = state_exact.rho
    vx_exact = state_exact.vx
    vt_exact = state_exact.vt

    # Plot
    axes[i, 0].plot(x_exact, P_exact, "k-", label="Exact")
    axes[i, 0].set_ylabel(f"P (v_t={v_t})")
    axes[i, 0].set_title("Pressure")
    axes[i, 0].legend()

    axes[i, 1].plot(x_exact, n_exact, "k-", label="Exact")
    axes[i, 1].set_ylabel(f"n (v_t={v_t})")
    axes[i, 1].set_title("Density")

    axes[i, 2].plot(x_exact, vx_exact, "k-", label="Exact")
    axes[i, 2].set_ylabel(f"vx (v_t={v_t})")
    axes[i, 2].set_title("Normal velocity")

    axes[i, 3].plot(x_exact, vt_exact, "k-", label="Exact")
    axes[i, 3].set_ylabel(f"vt (v_t={v_t})")
    axes[i, 3].set_title("Tangent velocity")

    print(f"\nv_t = {v_t}:")
    print(f"  Star state: P*={wavefan.states[1].pressure:.6f}, v_x*={wavefan.states[1].vx:.6f}")
    print(f"  Contact v_t*={wavefan.states[1].vt:.6f}")

    # Check v_t range
    print(f"  vt range: [{np.min(vt_exact):.4f}, {np.max(vt_exact):.4f}]")

for ax in axes[-1, :]:
    ax.set_xlabel("x")

plt.tight_layout()
plt.savefig("tangent_velocity_exact.png", dpi=150)
print("\nSaved tangent_velocity_exact.png")
