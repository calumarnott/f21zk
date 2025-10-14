import numpy as np

# Describe what this system expects and controls
STATE_NAMES = ["theta", "theta_dot"] # Names of state variables, in order
CONTROL_AXES = ["theta"] # Controlled axes. E.g. for a pendulum.

# Describe the required parameters and their types. Use separate
# dataclasses for hierarchical/nested parameters.
@dataclass
class Inner:
    mass: float
    position: np.ndarray

# --- Top-level dataclass that groups inner dataclass ---
# Must be named Params
@dataclass
class Params:
    mass: float
    length: float
    damping: float
    hierarchical_params: Inner

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

