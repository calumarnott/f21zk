# simulate_pendulum.py
"""
Simple simulator and plotting for the inverted pendulum.
Usage (as a script):
    python simulate_pendulum.py

You can edit the "config" section at the bottom to try different torques and initial states.

Still to do: 
 - export more states (including target theta and target theta_dot)
 - add more experiments with a variety of initial conditions to bulk out training data
 
 - perform adversarial attacks (PGD) on the NN controller to find failures
 - perform adversarial training to improve robustness and compare performance
 - perform reachability analysis to verify safety
 
"""
from typing import Callable, Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp
from matplotlib import animation
import torch
try:
    from matplotlib.animation import PillowWriter
except Exception:
    PillowWriter = None
from pendulum_dynamics import pendulum_rhs, PendulumParams

@dataclass
class SimConfig:
    t0: float = 0.0 # start time
    tf: float = 5.0 # end time
    dt: float = 0.05 # time step 
    x0: np.ndarray = None   # theta, theta_dot
    torque_limit: float = 5.0                            # optional saturation (N*m)

    def __post_init__(self):
        if self.x0 is None:
            self.x0 = np.array([1.1, 0.1], dtype=float)


def make_torque_fn(kind: str = "zero", Kp: float = 10.0, Kd: float = 2.0, theta_ref: float = 0.0, torque_limit: float = 5.0) -> Callable:
    """
    Returns u(t, x): torque function.
    - "zero": no actuation.
    - "pd":   PD about theta_ref (upright = 0). Not guaranteed to stabilize globally but useful to explore.
    """
    def clamp(v, lo, hi): 
        return max(lo, min(hi, v))

    if kind == "pd":
        def u(t, x):
            theta, theta_dot = float(x[0]), float(x[1])
            T = Kp*(theta_ref - theta) + Kd*(0.0 - theta_dot) # PD control
            return clamp(T, -torque_limit, torque_limit)
        return u
    elif kind == "nn":
        # Load a trained NN model (assumes model file exists)
        from models import SimpleNN
        model = SimpleNN(input_dim=2, hidden_dim=256, output_dim=1)
        model.load_state_dict(torch.load("data/pendulum_model.pth", weights_only=True))
        model.eval()
        def u_nn(t, x):
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0) # shape (1, 2)
                T_tensor = model(x_tensor) # shape (1, 1)
                T = float(T_tensor.item())
                return clamp(T, -torque_limit, torque_limit)
        return u_nn
    else:
        def u_zero(t, x):
            return 0.0
        return u_zero

def plots(df: pd.DataFrame):
    # 1) theta(t)
    plt.figure()
    plt.plot(df["t"].values, df["theta"].values)
    plt.xlabel("time [s]")
    plt.ylabel("theta [rad] (0 = upright)")
    plt.title("Inverted Pendulum: Angle vs Time")
    plt.grid(True)
    plt.tight_layout()

    # 2) theta_dot(t)
    plt.figure()
    plt.plot(df["t"].values, df["theta_dot"].values)
    plt.xlabel("time [s]")
    plt.ylabel("theta_dot [rad/s]")
    plt.title("Angular Rate vs Time")
    plt.grid(True)
    plt.tight_layout()

    # 3) Phase portrait
    plt.figure()
    plt.plot(df["theta"].values, df["theta_dot"].values)
    plt.xlabel("theta [rad]")
    plt.ylabel("theta_dot [rad/s]")
    plt.title("Phase Portrait")
    plt.grid(True)
    plt.tight_layout()

    # 4) Torque
    plt.figure()
    plt.plot(df["t"].values, df["torque"].values)
    plt.xlabel("time [s]")
    plt.ylabel("torque [N·m]")
    plt.title("Control Torque vs Time")
    plt.grid(True)
    plt.tight_layout()

def simulate_scipy(params: PendulumParams, sim: SimConfig, u_fn: Callable) -> pd.DataFrame:
    """
    Simulate using SciPy's solve_ivp (Runge-Kutta(5) Dormand–Prince).
    Resamples to fixed dt to match plotting/animation pipeline.
    """
    t_eval = np.arange(sim.t0, sim.tf + 1e-12, sim.dt)
    def rhs(t, x):
        return pendulum_rhs(t, x, u_fn, params)
    sol = solve_ivp(rhs, (sim.t0, sim.tf), sim.x0.astype(float), t_eval=t_eval, rtol=1e-8, atol=1e-10)
    # Compute torque time series at sampled grid
    U = np.array([float(u_fn(ti, np.array([th, thd]))) for ti, th, thd in zip(sol.t, sol.y[0], sol.y[1])])
    df = pd.DataFrame({"t": sol.t, "theta": sol.y[0], "theta_dot": sol.y[1], "torque": U})
    return df

def animate_rod(df, params, filename="pendulum_animation.mp4", fps=30, scale=1.2, out_dir: str | Path = "."):
    """
    Render the inverted pendulum as a rod+bob.
    - filename: ends with .mp4 (uses ffmpeg) or .gif (uses Pillow).
    - out_dir: directory to save into; created if it doesn't exist.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    ext = out_path.suffix.lower()

    L = params.L
    t = df["t"].values
    theta = df["theta"].values
    x =  L * np.sin(theta)
    y = L * np.cos(theta)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.2*L, 1.2*L)
    ax.set_ylim(-1.2*L, 1.2*L)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title("Inverted Pendulum Animation")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    rod_line, = ax.plot([0, x[0]], [0, y[0]], lw=3)
    bob, = ax.plot([x[0]], [y[0]], marker='o', markersize=12)
    pivot = ax.plot([0], [0], marker='o', markersize=6)[0]
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        rod_line.set_data([0, x[0]], [0, y[0]])
        bob.set_data([x[0]], [y[0]])
        time_text.set_text("t = 0.00 s")
        return rod_line, bob, pivot, time_text

    def update(i):
        rod_line.set_data([0, x[i]], [0, y[i]])
        bob.set_data([x[i]], [y[i]])
        time_text.set_text(f"t = {t[i]:.2f} s")
        return rod_line, bob, pivot, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init,
                                  interval=int(1000.0 / fps), blit=True)

    # Save: prefer MP4 if requested and ffmpeg exists, otherwise GIF via Pillow.
    if ext == ".mp4" and "ffmpeg" in animation.writers.list():
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata=dict(artist="D2AIR"), bitrate=2000)
        ani.save(out_path.as_posix(), writer=writer)
    else:
        if PillowWriter is None:
            plt.close(fig)
            raise RuntimeError(
                "MP4 requested but ffmpeg not available, and Pillow not installed for GIF.\n"
                "Install ffmpeg (conda-forge) for MP4 or Pillow for GIF."
            )
        if ext != ".gif":
            out_path = out_path.with_suffix(".gif")
        ani.save(out_path.as_posix(), writer=PillowWriter(fps=fps))

    plt.close(fig)
    return out_path.as_posix()

if __name__ == "__main__":
    # --- Config ---
    params = PendulumParams(m=0.5, L=0.5, c=0.0, g=1.0)
    sim = SimConfig(t0=0.0, tf=5.0, dt=0.05, x0=np.array([1.3, 0.1], dtype=float), torque_limit=5.0)
    torque = "nn" # Choose torque: "zero", "pd" or "nn"
    u = make_torque_fn(kind=torque, Kp=15.0, Kd=0.5, theta_ref=0.0, torque_limit=sim.torque_limit) # 15.Kp=15.0, Kd=3.5 

    # --- Run ---
    df = simulate_scipy(params, sim, u)

    # Save CSV for reproducibility
    df.to_csv(f"data/pendulum_simulation_{torque}.csv", index=False)
    animate_rod(df, params, filename=f"data/pendulum_animation_{torque}.mp4", fps=30, scale=1.2)

    # --- Plots ---
    # plots(df)
    # plt.show()
