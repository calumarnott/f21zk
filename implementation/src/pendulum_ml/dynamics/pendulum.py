import numpy as np
from .base import Params

# Describe what this system expects and controls
REQUIRED_PARAMS = {
    "m": float,
    "L": float,
    "c": float,
    "g": float
    } # Dictionary matching params in the dynamics.params section of the config file
AXES = ["theta"] # Controlled axes. E.g. for a pendulum.
STATE_NAMES = ["theta", "theta_dot"] # Names of state variables, in order

def validate_params(params):
    """ Validate that all required parameters are present at any level of nesting.
    
    Args:
        params (dict): parameters dictionary
    Raises:
        ValueError: if a required parameter is missing or has the wrong type
    Returns:
        bool: True if all required parameters are present
    """
    for key, value in REQUIRED_PARAMS.items():
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
        if isinstance(value, dict):
            if not isinstance(params[key], dict):
                raise ValueError(f"Parameter {key} should be a dictionary.")
            validate_params(params[key])
    return True

def error(axis: str, x: np.ndarray, setpoint: float) -> float:
    """ Compute error for a given axis.

    Args:
        axis (str): axis name
        x (np.ndarray): current state vector
        setpoint (float): desired setpoint value

    Returns:
        float: error value
    """
    # axis_index = AXES.index(axis)
    try:
        axis_index = AXES.index(axis)
    except ValueError:
        raise ValueError(f"Axis '{axis}' not found in AXES list.")
    
    return setpoint - x[axis_index]

def sample_x0(rng, dyn_cfg: dict) -> np.ndarray:
    """System default x0 sampler. Used only if config doesn't override.
    
    Args:
        rng: np.random.Generator instance
        dyn_cfg (dict): dynamics configuration dictionary
    """
    theta0 = rng.uniform(-np.pi, np.pi)  # angle
    theta_dot0 = rng.uniform(-1.0, 1.0)   # angular velocity
    return np.array([theta0, theta_dot0], dtype=float)


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