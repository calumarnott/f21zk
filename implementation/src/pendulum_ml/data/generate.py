from pathlib import Path
import numpy as np
import pandas as pd
from ..dynamics.base import euler_step, rk4_step, build_controllers_from_cfg,\
    validate_params, control_error, wrap_to_pi
from ..utils import import_system

def _resolve_initial_states(cfg, cps):
    """ Resolve initial states from config or dynamics module.

    Args:
        cfg (dict): config dictionary
        cps (module): dynamics module
    Raises:
        ValueError: if initial states are not properly specified
    Returns:
        list: list of initial states as np.ndarrays
    """
    init_states = cfg["data"].get("initial_state", None)
    n_trajectories = cfg["data"]["n_trajectories"]
    
    if init_states is None:
        # no initial states provided, try to get sampler from dynamics module
        if hasattr(cps, "sample_x0"):
            rng = np.random.default_rng(cfg.get("seed", 42)) # create RNG with optional seed
            init_states = [cps.sample_x0(rng, cfg["dynamics"]) for _ in range(n_trajectories)]
        else:
            raise ValueError("No initial states provided in config, and dynamics module does not implement 'sample_x0'.")
    else:
        # initial states provided in config
        if not isinstance(init_states, list) \
            or len(init_states) != n_trajectories \
            or not all([len(s)==len(cps.STATE_NAMES) for s in init_states]):
            raise ValueError(f"Initial states must be a list of {n_trajectories} states, each of size {len(cps.STATE_NAMES)}. Got: {init_states}")
        
    # Ensure all initial states are np.ndarrays
    init_states = [np.array(s, dtype=float) for s in init_states]
    return init_states

def _resolve_time_params(cfg):
    """ Resolve time step parameters from config.

    Args:
        cfg (dict): config dictionary
    Raises:
        ValueError: if time parameters are not properly specified
    Returns:
        tuple: (T, dt, control_dt, n_steps, n_ctrl_steps)
    """
    dyn = cfg["dynamics"]
    dt = float(dyn.get("dt", 0.01))          # simulation time step
    control_dt = float(dyn.get("control_dt", dt)) # controller update period (should be >= dt and a multiple of dt)
    assert control_dt >= dt and (control_dt / dt).is_integer(), "control_dt must be >= dt and a multiple of dt"
    
    # Get time horizon or total sim time
    T = cfg["data"].get("sim_time", None)  # total sim time in seconds
    H = cfg["data"].get("horizon", None)   # total number of steps
    
    if T is None and H is None:
        raise ValueError("Either 'sim_time' (seconds) or 'horizon' (steps) must be specified in config data section.")
    if T is not None and H is not None:
        raise ValueError("Only one of 'sim_time' (seconds) or 'horizon' (steps) should be specified in config data section, not both.")
    if T is None:
        T = H * dt
        
    n_steps = int(T / dt)                  # total number of steps
    n_ctrl_steps = int(control_dt / dt)    # number of steps between controller updates
    
    return T, dt, control_dt, n_steps, n_ctrl_steps

def _next_filename(out_dir):
    """ Generate the next available filename in a directory.

    Args:
        out_dir (Path): output directory

    Returns:
        str: next available filename, e.g. "traj_000.csv"
    """
    existing = [f.name for f in out_dir.glob("traj_*.csv")]
    if not existing:
        return "traj_000.csv"
    nums = [int(f[5:8]) for f in existing if f[5:8].isdigit()]
    next_num = max(nums) + 1 if nums else 0
    return f"traj_{next_num:03d}.csv"

def _add_initial_states_to_cfg(cfg, init_states):
    """ Add initial states to config for reference.

    Args:
        cfg (dict): configuration dictionary
        init_states (list): list of initial states as np.ndarrays

    Returns:
        dict: updated configuration dictionary with initial states added
    """
    cfg["data"] = dict(cfg.get("data", {}))  # ensure data section exists
    cfg["data"]["initial_state"] = [s.tolist() for s in init_states]  # convert np.ndarrays to lists
    return cfg

def simulate(cfg, out_dir="data/raw"):
    """ Simulate a PID trajectory.

    Args:
        cfg (dict): configuration dictionary
        out_dir (str, optional): output directory to save the trajectory. Defaults to "data/raw/pendulum".

    Returns:
        list: list containing the path to the saved trajectory CSV file
    """
    sys_name = cfg["system"]
    out = Path(out_dir) / sys_name
    out.mkdir(parents=True, exist_ok=True) # ensure output directory exists
    
    # Load system module from config
    cps = import_system(sys_name)

    # Dynamics params (dict) — validated by the system params class
    dyn = dict(cfg["dynamics"])            # shallow copy in case we mutate
    params = validate_params(cps.Params, dyn["params"])  # validate params dict
    
    # Build controllers per axis from config
    controllers = build_controllers_from_cfg(cfg["controller"], cps.CONTROL_AXES)
    for c in controllers.values():
        c.reset()  # reset controller state
        
    # Get x0 from config or from dynamics module sampler
    init_states = _resolve_initial_states(cfg, cps)
    cfg = _add_initial_states_to_cfg(cfg, init_states)  # add to cfg for reference    
    
    # Build integrator and time params
    step = rk4_step if cfg["dynamics"]["integrator"]=="rk4" else euler_step # integrator function
    
    T, dt, dt_ctrl, n_steps, n_ctrl_steps = _resolve_time_params(cfg)
        
    # Build output path 
    out_path = out / _next_filename(out)
    rows = []  
        
    for i in range(cfg["data"]["n_trajectories"]):
        traj_id = i
        t = 0.0  # current time
    
        # reset controller state for each trajectory
        for c in controllers.values():
            c.reset()  
            
        # get initial state and add it to output
        x0 = init_states[i]  # initial state for this trajectory
        x = x0.copy()  # current state

        control_steps_counter = 0  # counter for control steps
        
        # --- main simulation loop ---
        
        
        
        while t < T:
            
            # if cps has method step_simulation, use it
            if hasattr(cps, "step_simulation"):
                x, u_dict, err_dict = cps.step_simulation(x, t, controllers, params, step, dt)
            else:
                
                if control_steps_counter % n_ctrl_steps == 0:
                    # Time to compute new control inputs
                    
                    u_dict, err_dict = {}, {}
                    for axis, ctrl in controllers.items():
                        # if you have a time-varying reference, setpoints[axis] = reference(axis, t)
                        # TODO: extend to trajectory tracking error
                        sp = cfg["controller"]["pid"].get(axis, {}).get("setpoint", 0.0)
                        err = control_error(cps, axis, x, sp)
                        u = ctrl.update(err, dt_ctrl)    # derivative/integral use control_dt
                        u_dict[axis]  = float(u)
                
                control_steps_counter += 1
                        
                # Step the dynamics with current control inputs
                x = step(x, u_dict, cps.f, params, dt)
                x[0] = wrap_to_pi(x[0])  # wrap angle theta to [-pi, pi]

                
                for axis, ctrl in controllers.items():
                    sp = cfg["controller"]["pid"].get(axis, {}).get("setpoint", 0.0)
                    err = control_error(cps, axis, x, sp)
                    err_dict[axis] = float(err)
            
            t += dt
            # --- record a row aligned as (state_t, error_t, u_t) ---
            row = {
                "traj_id": traj_id,
                "t": t,
            }
            # states (ordered by STATE_NAMES)
            for i, name in enumerate(cps.STATE_NAMES):
                row[name] = float(x[i])
            # errors and controls per axis
            for err_axis in cps.INPUT_ERROR_AXES:
                row[f"error_{err_axis}"] = err_dict[err_axis]
                
            for ctrl_axis in cps.OUTPUT_CONTROL_AXES:
                row[f"u_{ctrl_axis}"] = u_dict[ctrl_axis]
                
            rows.append(row)
            
    # Write the whole run into a single CSV
    # Ensure column order: id, t, states..., errors..., controls...
    col_order = ["traj_id", "t"] + cps.STATE_NAMES + \
                [f"error_{a}" for a in cps.INPUT_ERROR_AXES] + \
                [f"u_{a}" for a in cps.OUTPUT_CONTROL_AXES]
    df = pd.DataFrame(rows, columns=col_order)
    df.to_csv(out_path, index=False)
    print(f"Saved trajectory to {out_path}")
    return str(out_path)