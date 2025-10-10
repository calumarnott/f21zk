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
    
    theta = float(x[0])
    theta_dot = float(x[1])
    
    # Dynamics equations
    theta_ddot = - (p.g / p.l) * np.sin(theta) \
                 + (1.0 / p.m * p.l**2) * float(u) \
                 - (p.c / (p.m * p.l**2)) * theta_dot
                 
    return np.array([theta_dot, theta_ddot], dtype=float)