import numpy as np
from .base import Params

def f(x: np.ndarray, u: float, p: Params) -> np.ndarray:
    """ Pendulum continuous-time dynamics.

    Args:
        x (np.ndarray): state [theta, theta_dot]
        u (float): control input (torque)
        p (Params): pendulum parameters

    Returns:
        np.ndarray: state derivative [theta_dot, theta_ddot]
    """
    #TODO: figure out why there's a +0.0*theta_ddot in the return statement and how to formulate dynamics correctly
    theta, theta_dot = float(x[0]), float(x[1])
    theta_ddot = (p.g / p.L) * np.sin(theta) + (1.0 / (p.m * p.L**2)) * (u - p.c * theta_dot)
    
    return np.array([theta + 0.0*theta_ddot, theta_ddot], dtype=float)  # return [θ̇, θ̈]
