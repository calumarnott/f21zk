import numpy as np
import torch
from .dynamics.base import rk4_step, wrap_to_pi

def generate_trajectory_from_NN_controller(cfg, cps, model):
    """ Generate a trajectory using a learned NN controller.

    Args:
        cfg (dict): Configuration dictionary containing simulation parameters.
        cps (module): dynamics module (e.g. pendulum_ml.dynamics.pendulum)
        model (torch.nn.Module): trained neural network controller
        x0 (np.ndarray): initial state
        T (float): total simulation time
        dt (float): simulation time step
        dt_ctrl (float): control time step

    Returns:
        np.ndarray: trajectory array of shape (num_steps, state_dim + num_control_axes)
    """
    model.eval()  # set model to evaluation mode
    dt = float(cfg["dynamics"].get("dt", 0.01))  # simulation time step
    dt_ctrl = float(cfg["dynamics"].get("control_dt", dt)) # controller
    x0 = np.array(cfg["data"]["initial_state"])
    T = float(cfg["data"]["sim_time"])
    assert x0.shape == (len(cps.STATE_NAMES),), f"Initial state shape mismatch. Expected {(len(cps.STATE_NAMES),)}, got {x0.shape}"
    
    num_steps = int(T / dt)
    n_ctrl_steps = int(dt_ctrl / dt)
    state_dim = len(cps.STATE_NAMES)
    num_controls = len(cps.CONTROL_AXES)
    params = cps.validate_params(cps.Params, cfg["dynamics"]["params"])
    
    
    trajectory = np.zeros((num_steps, state_dim + num_controls), dtype=float)
    x = x0.copy()
    t = 0.0
    control_steps_counter = 0
    
    with torch.no_grad():
        for i in range(num_steps):
            if control_steps_counter % n_ctrl_steps == 0:
                # Time to compute new control inputs
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # shape (1, state_dim)
                u_tensor = model(x_tensor)  # shape (1, num_controls)
                u = u_tensor.squeeze(0).numpy()  # shape (num_controls,)
            control_steps_counter += 1
            
            # Step the dynamics with current control inputs
            x = rk4_step(x, {axis: float(u[j]) for j, axis in enumerate(cps.CONTROL_AXES)}, cps.f, params, dt)
            x[0] = wrap_to_pi(x[0])  # wrap angle theta to [-pi, pi]
            t += dt
            
            trajectory[i, :state_dim] = x
            trajectory[i, state_dim:] = u
            
    return trajectory