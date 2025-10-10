from pathlib import Path
import numpy as np, pandas as pd
from ..dynamics.base import euler_step, rk4_step, build_controllers_from_cfg
import importlib


def _import_system(system_name: str):
    """ Import a dynamics module by system name.

    Args:
        system_name (str): system name, e.g. "pendulum"

    Raises:
        ModuleNotFoundError: _if the module cannot be found

    Returns:
        module: the imported dynamics module
    """
    # e.g. "pendulum" -> "src.pendulum_ml.dynamics.pendulum"
    # mod = importlib.import_module(f"src.pendulum_ml.dynamics.{system_name}")
    try:
        mod = importlib.import_module(f"pendulum_ml.dynamics.{system_name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Could not find dynamics module for system '{system_name}'. Ensure 'src/pendulum_ml/dynamics/{system_name}.py' exists.") from e
    return mod

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

def simulate(cfg, out_dir="data/raw/pendulum"):
    """ Simulate a PID trajectory.

    Args:
        cfg (dict): configuration dictionary
        out_dir (str, optional): output directory to save the trajectory. Defaults to "data/raw/pendulum".

    Returns:
        list: list containing the path to the saved trajectory CSV file
    """
    
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True) # ensure output directory exists
    
    # Load system module from config
    sys_name = cfg["system"]               # e.g., "pendulum"
    cps = _import_system(sys_name)

    # Dynamics params (dict) — validated by the system itself
    dyn = dict(cfg["dynamics"])            # shallow copy in case we mutate
    cps.validate_params(dyn["params"])  # validate params dict
    
    # Build controllers per axis from config
    controllers = build_controllers_from_cfg(cfg["controller"], cps.AXES)
    for c in controllers.values():
        c.reset()  # reset controller state
        
    # Get x0 from config or from dynamics module sampler
    init_states = cfg["data"].get("initial_state", None)
    n_trajectories = cfg["data"]["n_trajectories"]
    
    if init_states is None:
        # no initial states provided, try to get sampler from dynamics module
        if hasattr(cps, "sample_x0"):
            rng = np.random.default_rng(cfg.get("seed", None)) # create RNG with optional seed
            init_states = [cps.sample_x0(rng, dyn) for _ in range(n_trajectories)]
        else:
            raise ValueError("No initial states provided in config, and dynamics module does not implement 'sample_x0'.")
    else:
        # initial states provided in config
        if not isinstance(init_states, list) \
            or len(init_states) != n_trajectories\ 
            or not all([len(s)==cps.STATE_SIZE for s in init_states]):
                
            raise ValueError(f"Initial states must be a list of {n_trajectories} states, each of size {cps.STATE_SIZE}. Got: {init_states}")
        
    # Ensure all initial states are np.ndarrays
    init_states = [np.array(s, dtype=float) for s in init_states]
    
    
    
    # Build integrator and time params
    step = rk4_step if cfg["dynamics"]["integrator"]=="rk4" else euler_step # integrator function
    dt = float(dyn["dt"])
    control_dt = float(dyn.get("control_dt", dt)) # controller update period (should be >= dt and a multiple of dt)
    assert control_dt >= dt and (control_dt / dt).is_integer(), "control_dt must be >= dt and a multiple of dt"

    # Build output path 
    out_path = out / _next_filename(out)
    rows = []  

    for x0 in init_states:
        for c in controllers.values():
            c.reset()  # reset controller state for each trajectory

        x = x0.copy()  # current state
        
        # For convenience, pull per-axis setpoints from the controller cfg
        # (we stored them when building controllers; if not, use 0.0)
        setpoints = {}
        for axis, ctrl in controllers.items():
            sp = getattr(ctrl, "setpoint", 0.0)
            setpoints[axis] = float(sp)

        for k in range(T):
            # compute per-axis errors & controls at time t from current state x
            u_dict = {}
            err_dict = {}
            for axis, ctrl in controllers.items():
                err = cps.error(axis, x, setpoints[axis])
                u_axis = ctrl.update(err, dt)
                err_dict[axis] = float(err)
                u_dict[axis] = float(u_axis)

            # --- record a row aligned as (state_t, error_t, u_t) ---
            row = {
                "traj_id": traj_id,
                "t": t,
            }
            # states (ordered by STATE_NAMES)
            for i, name in enumerate(cps.STATE_NAMES):
                row[name] = float(x[i])

            # errors and controls per axis
            for axis in cps.AXES:
                row[f"error_{axis}"] = err_dict[axis]
                row[f"u_{axis}"] = u_dict[axis]

            rows.append(row)

            # --- integrate to next state using chosen integrator ---
            if integ in ("rk4", "runge", "runge-kutta"):
                x = rk4_step(x, u_dict, cps.f, dyn["params"])
            elif integ in ("euler", "forward-euler"):
                x = euler_step(x, u_dict, cps.f, dyn["params"])
            else:
                raise ValueError(f"Unknown integrator '{integ}'")
            t += dt

    # Write the whole run into a single CSV
    df = pd.DataFrame(rows)

    # Ensure column order: id, t, states..., errors..., controls...
    state_cols = list(cps.STATE_NAMES)
    err_cols = [f"error_{a}" for a in cps.AXES]
    u_cols = [f"u_{a}" for a in cps.AXES]
    df = df[["traj_id", "t"] + state_cols + err_cols + u_cols]

    df.to_csv(out_path, index=False)
    return [str(out_path)]
        
        
        
        
        
        
        
        
    # p = Params(**{k: cfg["dynamics"][k] for k in ("m","L","c","g","dt")}) # pendulum params
    
    # step = rk4_step if cfg["dynamics"]["integrator"]=="rk4" else euler_step # integrator function

    # # simple open-loop torque (baseline): zero torque
    # def torque_fn(x): 
    #     return 0.0

    # T = cfg["data"]["horizon"]
    # x = np.array([1.0, 0.0], dtype=float)  # initial [theta, theta_dot]
    # rows=[]
    
    # for t in range(T):
        
    #     u = float(torque_fn(x)) # torque at time t
        
    #     # record time, state, control
    #     rows.append({"t": t*p.dt, "theta": x[0], "theta_dot": x[1], "torque": u})
        
    #     # step dynamics
    #     x = step(x, u, dyn.f, p)
        
    # df = pd.DataFrame(rows) # create DataFrame
    
    # path = out / "traj_000.csv"
    
    # df.to_csv(path, index=False)
    
    return [path]
