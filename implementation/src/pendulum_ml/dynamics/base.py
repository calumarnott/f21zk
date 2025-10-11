import numpy as np

def euler_step(x, u, f, p):
    """ Simple Euler integration step.

    Args:
        x (float): current state
        u (float): current control input
        f (callable): dynamics function f(x,u,p)
        p (Params): pendulum parameters

    Returns:
        float: next state after one Euler step
    """
    return x + p.dt * f(x,u,p)

def rk4_step(x, u, f, p):
    """ Runge-Kutta 4th order integration step.

    Args:
        x (float): current state
        u (float): current control input
        f (callable): dynamics function f(x,u,p)
        p (Params): pendulum parameters

    Returns:
        float: next state after one RK4 step
    """
    k1 = f(x,u,p)
    k2 = f(x + 0.5*p.dt*k1, u, p)
    k3 = f(x + 0.5*p.dt*k2, u, p)
    k4 = f(x + p.dt*k3, u, p)
    return x + (p.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

class PIDController:
    """ Reusable, axis-agnostic PID with simple anti-windup (conditional integration)."""
    
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, u_min=None, u_max=None):
        self.Kp = float(Kp) # proportional gain
        self.Ki = float(Ki) # integral gain 
        self.Kd = float(Kd) # derivative gain
        self.u_min = u_min # saturation limits of actuator
        self.u_max = u_max
        
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        if dt <= 0:
            raise ValueError("dt must be > 0")

        # Proportional
        P = self.Kp * error
        
        # Integral
        I = self.Ki * self.integral

        # Derivative
        D = 0.0 if not self.prev_error else self.Kd * (error - self.prev_error) / dt

        # Unclamped control
        u_unclamped = P + I + D
        
        # Apply saturations -> actuator limits
        u = u_unclamped
        if self.u_min is not None:
            u = max(self.u_min, u)
        if self.u_max is not None:
            u = min(self.u_max, u)

        # TODO: Consult with calum
        """Calum's version:
        
        # Anti-windup: only integrate when not saturated 
        # OR when error drives output back toward range 
        if (self.u_min is None or u_unclamped > self.u_min or error < 0) and \
            (self.u_max is None or u_unclamped < self.u_max or error > 0): 
            self.integral += error * dt
        """
        
        # Anti-windup: commit integral only if not saturating in the "wrong" direction
        saturated_low  = (self.u_min is not None) and (u == self.u_min)
        saturated_high = (self.u_max is not None) and (u == self.u_max)

        commit_integral = True
        if saturated_low and (error < 0):    # would push further negative
            commit_integral = False
        if saturated_high and (error > 0):   # would push further positive
            commit_integral = False

        if commit_integral:
            self.integral = I  # commit
        # else: keep previous integral (no windup)

        self.prev_error = error
        
        return u
    
def build_controllers_from_cfg(ctrl_cfg: dict, axes: list[str]):
    """Build a dict of controllers keyed by axis name from YAML.
    
       ctrl_cfg looks like:
       { "type": "pid", "pid": { "theta": {...}, "x": {...}, ... } }
       
       # TODO: extend to inner-loop controllers, specified in config as:
       { "type": "pid", "pid": { "axes": {"theta": {...}, "x": {...}, ... }, 
                                 "pid" : {"axes": {...}}} }
    """
    if ctrl_cfg.get("type", "pid").lower() != "pid":
        raise NotImplementedError("Only 'pid' controller type is implemented right now.")

    pid_block = ctrl_cfg.get("pid", {})
    
    controllers = {}
    for axis in axes:
        p = pid_block.get(axis)
        if p is None:
            raise KeyError(f"Missing PID block for axis '{axis}' in controller.pid")
        controllers[axis] = PIDController(
            Kp=p.get("Kp", 0.0),
            Ki=p.get("Ki", 0.0),
            Kd=p.get("Kd", 0.0),
            u_min=p.get("u_min", None),
            u_max=p.get("u_max", None),
        )
        
    return controllers

def validate_params(cps, params):
    """ Validate that all required parameters are present at any level of nesting.
    
    Args:
        params (dict): parameters dictionary
    Raises:
        ValueError: if a required parameter is missing or has the wrong type
    Returns:
        bool: True if all required parameters are present
    """
    for key, value in cps.REQUIRED_PARAMS.items():
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
        if isinstance(value, dict):
            if not isinstance(params[key], dict):
                raise ValueError(f"Parameter {key} should be a dictionary.")
            validate_params(params[key])
    return True

def control_error(cps, axis: str, x: np.ndarray, setpoint: float) -> float:
    """ Compute error for a given axis.

    Args:
        cfs (module): dynamics module (e.g. pendulum_ml.dynamics.pendulum)
        axis (str): axis name
        x (np.ndarray): current state vector
        setpoint (float): desired setpoint value

    Returns:
        float: error value
    """
    assert axis in cps.CONTROL_AXES, f"Axis '{axis}' not in CONTROL_AXES {cps.CONTROL_AXES}"
    try:
        axis_index = cps.STATE_NAMES.index(axis)
    except ValueError:
        raise ValueError(f"Axis '{axis}' not found in the state vector.")
    
    return setpoint - x[axis_index]

def trajectory_error(cps, axis: str, x: np.ndarray, trajectory: np.ndarray, t: float, dt: float) -> float:
    """ Compute error for a given axis against a trajectory.

    Args:
        cps (module): dynamics module (e.g. pendulum_ml.dynamics.pendulum)
        axis (str): axis name
        x (np.ndarray): current state vector
        trajectory (np.ndarray): desired trajectory (num_steps x state_dim)
        t (float): current time
        dt (float): time step size
    Returns:
        float: error value
    """
    assert axis in cps.TRAJECTORY_AXES, f"Axis '{axis}' not in TRAJECTORY_AXES {cps.TRAJECTORY_AXES}"
    try:
        axis_index = cps.STATE_NAMES.index(axis)
    except ValueError:
        raise ValueError(f"Axis '{axis}' not found in the state vector.")
    
    step = int(t / dt)
    if step >= trajectory.shape[0]:
        step = trajectory.shape[0] - 1  # clamp to last step if out of bounds
    
    setpoint = trajectory[step, axis_index]
    return setpoint - x[axis_index]