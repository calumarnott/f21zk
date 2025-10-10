import numpy as np
from .base import Params

def f(state: np.ndarray, control: float, params: Params) -> np.ndarray:
    """ Pendulum continuous-time dynamics.

    Args:
        state (np.ndarray): state [theta, theta_dot]
        control (float): control input (torque)
        params (Params): pendulum parameters

    Returns:
        np.ndarray: state derivative [theta_dot, theta_ddot]
    """
    
    theta = float(state[0])
    theta_dot = float(state[1])
    
    # Dynamics equations
    # theta_ddot = - (g / l) * sin(theta) + (1/m/l^2) * u - (c/m/l^2) * theta_dot
    theta_ddot = - (params.g / params.l) * np.sin(theta) \
                 + (1.0 / params.m * params.l**2) * float(control) \
                 - (params.c / (params.m * params.l**2)) * theta_dot
                 
    return np.array([theta_dot, theta_ddot], dtype=float)