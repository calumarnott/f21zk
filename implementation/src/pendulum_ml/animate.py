import numpy as np
import torch
from .dynamics.base import rk4_step, wrap_to_pi, control_error, build_controllers_from_cfg
from .data.process import first_window
from .utils import import_system

def generate_trajectory_from_NN_controller(cfg, cps, model, initial_states):
    """ Generate a trajectory using a learned NN controller.

    Args:
        cfg (dict): Configuration dictionary containing simulation parameters.
        cps (module): dynamics module (e.g. pendulum_ml.dynamics.pendulum)
        model (torch.nn.Module): trained neural network controller
        initial_states (list or np.ndarray): list or array of initial states, shape (num_trajectories, state_dim)

    Returns:
        np.ndarray: trajectory array of shape (num_steps, state_dim + num_control_axes)
    """
    model.eval()  # set model to evaluation mode
    dt = float(cfg["dynamics"].get("dt", 0.01))  # simulation time step
    dt_ctrl = float(cfg["dynamics"].get("control_dt", dt)) # controller
    if initial_states is None:
        raise ValueError("initial_states must be provided.")
    init_states = np.array(initial_states) # list of initial states
    T = float(cfg["data"]["sim_time"])
    assert init_states.shape[1] == len(cps.STATE_NAMES), f"Initial state shape mismatch. Expected {len(cps.STATE_NAMES)}, got {init_states.shape[1]}"
    
    num_steps = int(T / dt)
    n_ctrl_steps = int(dt_ctrl / dt)
    state_dim = len(cps.STATE_NAMES)
    num_controls = len(cps.CONTROL_AXES)
    num_err_inputs = len(cps.INPUT_ERROR_AXES)
    params = cps.validate_params(cps.Params, cfg["dynamics"]["params"])


    trajectory = np.zeros((num_steps * len(init_states), 1 + state_dim + num_err_inputs), dtype=float)

    t = 0.0
    control_steps_counter = 0
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    
    controllers = build_controllers_from_cfg(cfg["controller"], cps.CONTROL_AXES)
    for c in controllers.values():
        c.reset()
    
    with torch.no_grad():
        for k in range(len(init_states)): # for each init state
            x0 = init_states[k]
            x = x0.copy()
            
            for c in controllers.values():
                c.reset()
                
            err_dict = {}
            
            for axis, ctrl in controllers.items():
                sp = cfg["controller"]["pid"].get(axis, {}).get("setpoint", 0.0)
                err_dict[axis] = control_error(cps, axis, x, sp)

            x_input = np.concatenate([x, [err_dict[axis] for axis in cps.INPUT_ERROR_AXES]], axis=0)
            t = 0.0
            x_traj = np.concatenate([[t], x_input], axis=0)
            
            # if window in cfg, x_input = first_window(...)
            if cfg["data"].get("window", {}).get("length", 0) > 0:
                x_input, _ = first_window(x_input, None, cfg["data"]["window"]["length"])
                x_input = x_input.T # shape (state_dim, window_length)

            trajectory[k*num_steps] = x_traj

            for i in range(1, num_steps):
                if control_steps_counter % n_ctrl_steps == 0:
                    # Time to compute new control inputs
                    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)  # shape (1, input_dim)
                    # to device
                    x_tensor = x_tensor.to(device)
                    u_tensor = model(x_tensor)  # shape (1, num_output_controls)
                    u = u_tensor.squeeze(0).cpu().numpy()  # shape (num_output_controls,)
                    
                    

                    ### try calling controllers to test nn controller vs pid behavior
                    # x_pid, u_dict_pid, err_dict_pid = cps.step_simulation(x_pid, t, controllers, params, rk4_step, dt)
                    

                control_steps_counter += 1
                
                for j, axis in enumerate(cps.OUTPUT_CONTROL_AXES):
                    # clamp control inputs to actuator limits if specified in cfg
                    u_min = cfg["controller"].get("pid", {}).get(axis, {}).get("u_min", None)
                    u_max = cfg["controller"].get("pid", {}).get(axis, {}).get("u_max", None)
                    if u_min is not None and u[j] < u_min:
                        u[j] = u_min
                    if u_max is not None and u[j] > u_max:
                        u[j] = u_max

                # Step the dynamics with current control inputs
                x = rk4_step(x, {axis: float(u[j]) for j, axis in enumerate(cps.OUTPUT_CONTROL_AXES)}, cps.f, params, dt)
                
                

                for l, name in enumerate(cps.STATE_NAMES):
                    if "angle" in name or "theta" in name or "phi" in name:
                        x[l] = wrap_to_pi(x[l])

                t += dt
                    
                for axis, ctrl in controllers.items():
                    sp = cfg["controller"]["pid"].get(axis, {}).get("setpoint", 0.0)
                    err = control_error(cps, axis, x, sp)
                    err_dict[axis] = float(err)
                    
                if cfg["data"].get("window", {}).get("length", 0) > 0:
                    # copy of tensor
                    x_input_window = x_input

                x_input = np.concatenate([x, [err_dict[axis] for axis in cps.INPUT_ERROR_AXES]], axis=0)
                
                x_traj = np.concatenate([[t], x_input], axis=0)

                if cfg["data"].get("window", {}).get("length", 0) > 0:
                    x_input_window = np.roll(x_input_window, shift=-1, axis=1)
                    x_input_window[:, -1] = new_column = x_input
                    x_input = x_input_window

                trajectory[k*num_steps + i] = x_traj  # state + error

    return trajectory