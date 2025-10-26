import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from ..dynamics.base import validate_params

STATE_NAMES = ["x", "z", "theta", "xq_dot", "zq_dot",\
                "theta_dot", "l", "phi", "l_dot", "phi_dot"]
# CONTROL_AXES = ["x", "z", "theta", "phi", "l"]
CONTROL_AXES = ["x", "z", "theta", "phi"]


# --- Leaf-level dataclasses ---
@dataclass
class QuadParams:
    mass: float
    inertia: float
    arm_length: float

@dataclass
class PayloadParams:
    mass: float
    rope_length: float

@dataclass
class EnvironmentParams:
    gravity: float

@dataclass
class WinchParams:
    model: str
    omega: float = 0.0
    
@dataclass
class ActuatorParams:
    max_thrust: float
    max_pitch_torque: float
    min_pitch_torque: float

# --- Top-level dataclass that groups them ---
@dataclass
class Params:
    quad: QuadParams
    payload: PayloadParams
    environment: EnvironmentParams
    winch: WinchParams
    actuators: ActuatorParams



# cps.step_simulation(x, t, controllers, cfg, params, dt, control_steps_counter, n_ctrl_steps)

def step_simulation(state: np.ndarray, t: float, controllers: dict, params: Params, step: callable,
                    dt: float) -> (np.ndarray, dict, dict):
    """ Step the simulation by one time step.

    Args:
        state (np.ndarray): current state vector
        t (float): current time
        controllers (dict): dictionary of controller instances for each control axis
        cfg (dict): configuration dictionary
        params (Params): parameters dataclass instance
        dt (float): time step size
    Returns:
        np.ndarray: next state vector
        dict: control inputs
        dict: control errors
    """
    x_q, z_q, theta, xdot, zdot, thetadot, l, phi, ldot, phidot = state
    
    u_dict = {}
    err_dict = {}
    
    
    # Desired references
    x_ref, z_ref = 4.0, 5.0  # initial position
    phi_ref = 0.0            # desired payload angle (rad)
    
    
    # --- Quadcopter Position Control ---
    a_x_pos = controllers["x"].update(x_ref - x_q, dt) 
    a_z_des = controllers["z"].update(z_ref - z_q, dt)
    err_dict["x"] = float(x_ref - x_q)
    
    # --- Payload Swing Control ---
    a_x_swing = controllers["phi"].update(phi_ref - phi, dt)

    # Total desired lateral acceleration
    a_x_des = a_x_pos + a_x_swing # sum contributions to lat. acceleration from position and swing controllers
    
    
    # Desired pitch from lateral accel
    theta_des = -np.arctan2(a_x_des, max(params.environment.gravity, 1e-6)) 
    
    # Attitude control
    tau_des = controllers["theta"].update(theta_des - theta, dt)

    u_dict["x"] = float(a_x_pos)
    u_dict["z"] = float(a_z_des)
    u_dict["theta"] = float(tau_des)
    u_dict["phi"] = float(a_x_swing)
    # u_dict["l"] = 0.0  # no winch control for now

    # Integrate dynamics
    state = step(state, u_dict, f, params, dt)
    
    err_dict["x"] = float(x_ref - x_q)
    err_dict["z"] = float(z_ref - z_q)
    err_dict["theta"] = float(theta_des - theta)
    err_dict["phi"] = float(phi_ref - phi)
    # err_dict["l"] = 0.0


    return state, u_dict, err_dict

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
    
    if out_path is None:
        raise ValueError("Missing argument for animate: out_path")
    
    if isinstance(trajectory_path, (str, Path)):
        trajectory_path = Path(trajectory_path)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
        # Load trajectory ignoring header and first column (traj_id)
        data = np.loadtxt(trajectory_path, delimiter=",", skiprows=1, 
                        usecols=range(1, 2 + len(STATE_NAMES) + len(CONTROL_AXES)))
    elif isinstance(trajectory_path, np.ndarray):
        data = trajectory_path
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
    else:
        raise ValueError("trajectory_path should be a file path or a numpy array.")
        
    params = validate_params(Params, cfg["dynamics"]["params"])
    
    d = params.quad.arm_length
    
    # Extract trajectories
    time = data[:, 0]
    quad_traj = data[:, 1:3]  # xq, zq
    theta_hist = data[:, 3]  # theta
    l_hist = data[:, 7]      # l
    phi_hist = data[:, 8]    # phi
        

    # Figure setup
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.grid(True)
    ax.set_title("Quadcopter + Payload Animation")

    # Axis limits
    margin = 2.0
    xmin, xmax = quad_traj[:, 0].min() - margin, quad_traj[:, 0].max() + margin
    zmin, zmax = quad_traj[:, 1].min() - margin, quad_traj[:, 1].max() + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)

    # Artists
    quad_body, = ax.plot([], [], lw=2.5, color='k')
    arm_line, = ax.plot([], [], lw=2, color='gray')
    rotor_L, = ax.plot([], [], 'o', color='C0', ms=8)
    rotor_R, = ax.plot([], [], 'o', color='C0', ms=8)
    rope_line, = ax.plot([], [], color='brown', lw=1.5)
    payload, = ax.plot([], [], 'o', color='red', ms=10)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # Geometry constants
    body_w, body_h = 3*d, 2*d
    arm_half = 4*d

    # --- Initialization ---
    def init():
        quad_body.set_data([], [])
        arm_line.set_data([], [])
        rotor_L.set_data([], [])
        rotor_R.set_data([], [])
        rope_line.set_data([], [])
        payload.set_data([], [])
        time_text.set_text("")
        return quad_body, arm_line, rotor_L, rotor_R, rope_line, payload, time_text

    # --- Update function ---
    def update(i):
        xq, zq = quad_traj[i]
        phi = phi_hist[i]
        l = l_hist[i]
        theta = theta_hist[i]
        xp = xq + l * np.sin(phi)
        zp = zq - l * np.cos(phi)

        # Rotation matrix (body frame to world)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Body rectangle (local coords)
        corners = np.array([
            [-body_w/2, -body_h/2],
            [ body_w/2, -body_h/2],
            [ body_w/2,  body_h/2],
            [-body_w/2,  body_h/2],
            [-body_w/2, -body_h/2],
        ])
        corners_rot = (R @ corners.T).T + np.array([xq, zq])
        quad_body.set_data(corners_rot[:, 0], corners_rot[:, 1])

        # Arms
        arm_pts = np.array([[-arm_half, 0], [arm_half, 0]])
        arm_rot = (R @ arm_pts.T).T + np.array([xq, zq])
        arm_line.set_data(arm_rot[:, 0], arm_rot[:, 1])
        rotor_L.set_data([arm_rot[0, 0]], [arm_rot[0, 1]])
        rotor_R.set_data([arm_rot[1, 0]], [arm_rot[1, 1]])


        # Rope and payload
        rope_line.set_data([xq, xp], [zq, zp])
        payload.set_data([xp], [zp])

        time_text.set_text(f"t = {time[i]:.2f} s")
        return quad_body, arm_line, rotor_L, rotor_R, rope_line, payload, time_text

    dt = cfg["dynamics"].get("dt", 0.01)  # default dt if not specified
    # --- Animation ---
    anim = animation.FuncAnimation(
        fig, update, frames=len(time),
        init_func=init, blit=True, interval=1000*dt
    )

    # Save or display
    print(f'Saving animation to {out_path} ...')
    if out_path.suffix.lower() == ".mp4":
        anim.save(out_path, writer="ffmpeg", fps=1./dt)
        print(f'Animation saved to {out_path}')
    elif out_path.suffix.lower() == ".gif":
        anim.save(out_path, writer="pillow", fps=1./dt)
        print(f'Animation saved to {out_path}')
    else:
        print(f"Unsupported extension: {out_path.suffix}, showing instead.")
        plt.show()

    plt.close(fig)
    
    
    # Optional plotting of state variables over time
    if plot:
        num_trajectories = cfg["data"].get("n_trajectories", 1)
        
        # first plot all trajectories in one figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        time = data[:, 0].reshape(num_trajectories, -1)
        x = data[:, 1].reshape(num_trajectories, -1)
        z = data[:, 2].reshape(num_trajectories, -1)
        phi = data[:, 8].reshape(num_trajectories, -1)
        
        for i in range(num_trajectories):
            if i == 0:
                axs[0].plot(time[i], x[i], label='x (position)', color='blue')
                axs[1].plot(time[i], z[i], label='z (altitude)', color='blue')
                axs[2].plot(time[i], phi[i], label='phi (payload angle)', color='blue')
            else:
                axs[0].plot(time[i], x[i], color='blue')
                axs[1].plot(time[i], z[i], color='blue')
                axs[2].plot(time[i], phi[i], color='blue')

        # x vs time
        axs[0].axhline(y=4.0, color='orange', linestyle='--', label='x setpoint')
        axs[0].set_title('x vs Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('x (m)')
        axs[0].legend()
        axs[0].grid()
        
        # z vs time
        axs[1].axhline(y=5.0, color='orange', linestyle='--', label='z setpoint')
        axs[1].set_title('z vs Time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('z (m)')
        axs[1].legend()
        axs[1].grid()
        
        # phi vs time
        axs[2].axhline(y=0.0, color='orange', linestyle='--', label='phi setpoint')
        axs[2].set_title('phi vs Time')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('phi (rad)')
        axs[2].legend()
        axs[2].grid()
            
        plt.tight_layout()
        # output path wihout .mp4 or .gif + _all_trajectories.png
        plot_path = out_path.parent / f"{out_path.stem}_all_trajectories.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"All trajectories plot saved to {plot_path}")
        
        
        # then plot each trajectory separately

        time = data[:, 0].reshape(num_trajectories, -1)
        x = data[:, 1].reshape(num_trajectories, -1)
        z = data[:, 2].reshape(num_trajectories, -1)
        phi = data[:, 8].reshape(num_trajectories, -1)
        
        for i in range(num_trajectories):
            fig, axs = plt.subplots(3, 1, figsize=(8, 12))
            
            # x vs time
            axs[0].plot(time[i], x[i], label='x (position)', color='blue')
            axs[0].axhline(y=4.0, color='orange', linestyle='--', label='x setpoint')
            axs[0].set_title('x vs Time')
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel('x (m)')
            axs[0].legend()
            axs[0].grid()
            
            # z vs time
            axs[1].plot(time[i], z[i], label='z (altitude)', color='blue')
            axs[1].axhline(y=5.0, color='orange', linestyle='--', label='z setpoint')
            axs[1].set_title('z vs Time')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('z (m)')
            axs[1].legend()
            axs[1].grid()
            
            # phi vs time
            axs[2].plot(time[i], phi[i], label='phi (payload angle)', color='blue')
            axs[2].axhline(y=0.0, color='orange', linestyle='--', label='phi setpoint')
            axs[2].set_title('phi vs Time')
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel('phi (rad)')
            axs[2].legend()
            axs[2].grid()
            
            plt.tight_layout()
            plot_path = out_path.parent / f"{out_path.stem}_{i+1}_plots.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot {i+1} saved to {plot_path}")
    
    return str(out_path)

# Unit vectors based on rope swing angle
def rope_vectors(phi):
    """Return unit vector along rope (quad -> payload) and tangential vector."""
    u = np.array([np.sin(phi), -np.cos(phi)])  # along rope
    t = np.array([np.cos(phi), np.sin(phi)])  # tangential, direction of phi+
    return u, t

# Dynamics function
def f(state: np.ndarray, control: dict, params: Params) -> np.ndarray:
    """
    Compute state derivatives for quad + rope payload system in 2D.

    State vector:
      [xq, zq, theta, xq_dot, zq_dot, theta_dot, l, phi, l_dot, phi_dot]
    Control vector:
      u_dict 
    
    Returns:
        dx: time derivative of state vector (10,)
    """
    # Unpack state
    xq, zq, theta, xq_dot, zq_dot, theta_dot, l, phi, l_dot, phi_dot = state

    a_x_pos = control["x"]
    a_z_des = control["z"]
    tau_des = control["theta"]
    a_x_swing = control["phi"]
    # u_l = control["l"]
    u_l = 0.0  # no winch control for now

    # --- Mixer ---
    # Desired thrust magnitude (vertical control)
    T_total_des = params.quad.mass * (params.environment.gravity + a_z_des) / max(np.cos(theta), 0.1)
    T_total_des = np.clip(T_total_des, 0.0, params.actuators.max_thrust * 4.0)

    pair_front = 0.5 * T_total_des + 0.5 * tau_des / max(params.quad.arm_length, 1e-6)
    pair_rear  = 0.5 * T_total_des - 0.5 * tau_des / max(params.quad.arm_length, 1e-6)
    T1 = T2 = 0.5 * pair_front
    T3 = T4 = 0.5 * pair_rear

    T1 = np.clip(T1, 0.0, params.actuators.max_thrust)
    T2 = np.clip(T2, 0.0, params.actuators.max_thrust)
    T3 = np.clip(T3, 0.0, params.actuators.max_thrust)
    T4 = np.clip(T4, 0.0, params.actuators.max_thrust)
    
    # Recompute totals (for consistency)
    T_total = T1 + T2 + T3 + T4
    tau = params.quad.arm_length * ((T1 + T2) - (T3 + T4))

    # Params
    m_q = params.quad.mass
    I_q = params.quad.inertia
    d = params.quad.arm_length

    m_p = params.payload.mass

    g = params.environment.gravity

    winch_model = params.winch.model
    omega_l = params.winch.omega

    # Rotor thrust totals
    T_tot = T1 + T2 + T3 + T4
    tau   = d * ((T1 + T2) - (T3 + T4))   # pitch torque

    # Quad thrust in world frame
    F_thrust = np.array([-T_tot * np.sin(theta),
                          T_tot * np.cos(theta)])

    # Preliminary quad acceleration without rope force
    acc_quad = F_thrust / m_q + np.array([0.0, -g])

    # Rope unit vectors
    u_vec, t_vec = rope_vectors(phi)

    # Winch dynamics
    if winch_model == "algebraic":
        l_ddot = 0.0
        l_dot_cmd = u_l
    elif winch_model == "first_order":
        l_dot_cmd = l_dot
        l_ddot = omega_l * (u_l - l_dot)
    else:
        raise ValueError("Unknown winch model")

    # --- Close the algebraic loop: iterate tension once ---
    # Iteration 1: use current acc_quad to get phi_ddot and T
    phi_ddot = (-g * np.sin(phi) + acc_quad.dot(t_vec) - 2.0 * l_dot * phi_dot) / max(l, 1e-6)
    Tension  = m_p * (g * np.cos(phi) + acc_quad.dot(u_vec) - (l_ddot - l * phi_dot**2))
    Tension  = max(Tension, 0.0)  # rope can't push

    # Update quad acceleration with rope feedback
    acc_quad = F_thrust / m_q + np.array([0.0, -g]) + (Tension / m_q) * u_vec

    # Optional: one more pass (helps a lot for bigger dt or larger swings)
    phi_ddot = (-g * np.sin(phi) + acc_quad.dot(t_vec) - 2.0 * l_dot * phi_dot) / max(l, 1e-6)
    Tension  = m_p * (g * np.cos(phi) + acc_quad.dot(u_vec) - (l_ddot - l * phi_dot**2))
    Tension  = max(Tension, 0.0)
    acc_quad = F_thrust / m_q + np.array([0.0, -g]) + (Tension / m_q) * u_vec

    # Quad pitch dynamics
    theta_ddot = tau / I_q

    # Assemble derivatives
    dx = np.zeros_like(state)
    dx[0] = xq_dot
    dx[1] = zq_dot
    dx[2] = theta_dot
    dx[3] = acc_quad[0]
    dx[4] = acc_quad[1]
    dx[5] = theta_ddot
    dx[6] = l_dot_cmd
    dx[7] = phi_dot
    dx[8] = l_ddot
    dx[9] = phi_ddot
    return dx