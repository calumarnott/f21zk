# pendulum_dynamics.py
"""
Inverted pendulum (torque at the pivot) continuous-time dynamics.

Angle theta is measured relative to the upward vertical (theta=0 is upright).
Equation (from coursework brief):
    theta_ddot = (g/L) * sin(theta) + (1/(m*L**2)) * (T - c*theta_dot)

State: x = [theta, theta_dot]
Control: T (torque, N*m), provided by a user-specified function u(t, x) in the simulator.

Typical params (ARCH-COMP style):
    m = 0.5, L = 0.5, c = 0.0, g = 1.0
"""
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

@dataclass
class PendulumParams:
    m: float = 0.5     # kg
    L: float = 0.5     # m
    c: float = 0.0     # N*m*s (viscous) What is c here?
    g: float = -9.81     # m/s^2 (non-dimensionalized gravity in spec) Should be -9.81 for real gravity?

def pendulum_rhs(t: float, x: Sequence[float], u: Callable[[float, np.ndarray], float], p: PendulumParams) -> np.ndarray:
    """
    Continuous-time RHS: xdot = f(t, x)
    """
    theta, theta_dot = float(x[0]), float(x[1])
    T = float(u(t, np.array([theta, theta_dot], dtype=float))) if u is not None else 0.0 # torque
    # Dynamics
    theta_ddot = (p.g / p.L) * np.sin(theta) + (1.0 / (p.m * p.L**2)) * (T - p.c * theta_dot)
    return np.array([theta_dot, theta_ddot], dtype=float)
