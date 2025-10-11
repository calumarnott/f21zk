import numpy as np

# Describe what this system expects and controls
REQUIRED_PARAMS = {} # Dictionary matching params in the dynamics section of the config file
STATE_NAMES = ["theta", "theta_dot"] # Names of state variables, in order
CONTROL_AXES = ["theta"] # Controlled axes. E.g. for a pendulum.


def sample_x0(rng, dyn_cfg: dict) -> np.ndarray:
    """ Sample an initial state.

    Args:
        rng: np.random.Generator instance
        dyn_cfg (dict): dynamics configuration dictionary
    """
    raise NotImplementedError("State sampler 'sample_x0' must be implemented in dynamics/<system_name>.py")


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

