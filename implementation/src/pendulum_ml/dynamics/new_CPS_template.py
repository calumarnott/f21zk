import numpy as np

# Describe what this system expects and controls
REQUIRED_PARAMS = {} # Dictionary matching params in the dynamics section of the config file
AXES = ["theta"] # Controlled axes. E.g. for a pendulum.

def validate_params(params):
    """ Validate that all required parameters are present at any level of nesting."""
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

def f(state: np.ndarray, control: np.ndarray, params: dict) -> np.ndarray:
    """ Dynamics function.

    Args:
        state (np.ndarray): current state vector
        control (np.ndarray): control input vector
        params (dict): parameters dictionary

    Returns:
        np.ndarray: state derivative vector
    """
    raise NotImplementedError("Dynamics function 'f' must be implemented in dynamics/<system_name>.py")

