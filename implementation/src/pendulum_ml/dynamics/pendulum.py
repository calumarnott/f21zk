from pathlib import Path
import numpy as np
from dataclasses import dataclass, fields
from ..dynamics.base import validate_params
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CONTROL_AXES = ["theta"] # Controlled axes. E.g. for a pendulum.
STATE_NAMES = ["theta", "theta_dot"] # Names of state variables, in order

@dataclass
class Params:
    m: float  # mass (kg)
    l: float  # length (m)
    c: float  # damping coefficient (N*m*s)
    g: float  # gravity (m/s^2)


def sample_x0(rng, dyn_cfg: dict) -> np.ndarray:
    """System default x0 sampler. Used only if config doesn't override.
    
    Args:
        rng: np.random.Generator instance
        dyn_cfg (dict): dynamics configuration dictionary
    """
    theta0 = rng.uniform(-np.pi, np.pi)  # angle
    theta_dot0 = rng.uniform(-1.0, 1.0)   # angular velocity
    return np.array([theta0, theta_dot0], dtype=float)

def animate(cfg, trajectory_path, out_path=None, plot=False) -> str:
    """ Create animation of a trajectory in mp4 format.

    Args:
        cfg (dict): config dictionary
        trajectory_path (str, Path or np.ndarray): path to trajectory CSV file or trajectory array
        out_path (str or Path, optional): path to save the output animation file. Defaults to "data/raw/file.mp4".
        plot (bool, optional): whether to generate plots of state variables over time. Defaults to False.

    Returns:
        str: path to the output animation file
    """
    if plot:
        raise NotImplementedError("Plotting state variable graphs is not yet implemented for pendulum system.")
    
    if out_path is None:
        raise ValueError("Missing argument for animate: out_path")
    
    if isinstance(trajectory_path, (str, Path)):
        trajectory_path = Path(trajectory_path)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
        # Load trajectory ignoring header and first two columns (traj_id, time)
        data = np.loadtxt(trajectory_path, delimiter=",", skiprows=1, 
                        usecols=range(1, 2 + len(STATE_NAMES) + len(CONTROL_AXES)))
    elif isinstance(trajectory_path, np.ndarray):
        data = trajectory_path
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
    else:
        raise ValueError("trajectory_path should be a file path or a numpy array.")
        
    params = validate_params(Params, cfg["dynamics"]["params"])

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.set_title("Pendulum Animation")
    
    # Axis limits
    margin = .5
    xmin, xmax = -params.l - margin, params.l + margin
    ymin, ymax = -params.l - margin, params.l + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Artists
    base, = ax.plot(0, 0, 'o', color='k', ms=8)
    rope_line, = ax.plot([], [], '-', color='brown', lw=1.5)
    payload, = ax.plot([], [], 'o', color='red', ms=10)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    error_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    
    def init():
        rope_line.set_data([], [])
        payload.set_data([], [])
        time_text.set_text('')
        error_text.set_text('')
        return rope_line, payload, time_text, error_text
    
    def update(frame):
        theta = data[frame, 1]  # theta
        x = params.l * np.sin(theta)
        y = -params.l * np.cos(theta)
        rope_line.set_data([0, x], [0, y])
        payload.set_data([x], [y])
        time_text.set_text(f'Time: {data[frame,0]:.2f} s')
        error_text.set_text(f'Error: {data[frame, -1]:.2f}')  # last column is error
        return rope_line, payload, time_text, error_text
    
    dt = cfg["dynamics"].get("dt", 0.01)  # default dt if not specified
    anim = animation.FuncAnimation(fig, update, 
                                      frames=len(data), 
                                      init_func=init, 
                                      interval=dt*1000, blit=True)
    anim.save(out_path, fps=1./dt, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)
    return str(out_path)
    
    

def f(state: np.ndarray, control: dict, params) -> np.ndarray:
    """ Pendulum continuous-time dynamics.

    Args:
        state (np.ndarray): state [theta, theta_dot]
        control (dict): control input dictionary {'axis': value}
        params (Params): pendulum parameters

    Returns:
        np.ndarray: state derivative [theta_dot, theta_ddot]
    """
    
    theta = float(state[0])
    theta_dot = float(state[1])
    
    # Dynamics equations
    # theta_ddot = - (g / l) * sin(theta) + (1/m/l^2) * u - (c/m/l^2) * theta_dot
    theta_ddot = - (params.g / params.l) * np.sin(theta) \
                 + (1.0 / params.m * params.l**2) * float(control["theta"]) \
                 - (params.c / (params.m * params.l**2)) * theta_dot
                 
    return np.array([theta_dot, theta_ddot], dtype=float)