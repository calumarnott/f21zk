from pathlib import Path
import numpy as np, pandas as pd
from ..dynamics.base import Params, euler_step, rk4_step
from ..dynamics import pendulum as dyn

# TODO: figure out how this works

def simulate(cfg, out_dir="data/raw/pendulum"):
    """ Simulate a single pendulum trajectory using simple open-loop torque (zero torque).

    Args:
        cfg (dict): configuration dictionary
        out_dir (str, optional): output directory to save the trajectory. Defaults to "data/raw/pendulum".

    Returns:
        list: list containing the path to the saved trajectory CSV file
    """
    
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True) # ensure output directory exists
    
    p = Params(**{k: cfg["dynamics"][k] for k in ("m","L","c","g","dt")}) # pendulum params
    
    step = rk4_step if cfg["dynamics"]["integrator"]=="rk4" else euler_step # integrator function

    # simple open-loop torque (baseline): zero torque
    def torque_fn(x): return 0.0

    T = cfg["data"]["horizon"]
    x = np.array([1.0, 0.0], dtype=float)  # initial [theta, theta_dot]
    rows=[]
    
    for t in range(T):
        
        u = float(torque_fn(x))
        rows.append({"t": t*p.dt, "theta": x[0], "theta_dot": x[1], "torque": u})
        x = step(x, u, dyn.f, p)
    df = pd.DataFrame(rows)
    path = out / "traj_000.csv"
    df.to_csv(path, index=False)
    return [path]
