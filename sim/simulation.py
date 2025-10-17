import numpy as np
import matplotlib.pyplot as plt
from dynamics import load_params, derivatives
from pid_control import PIDController
from pathlib import Path
from animation import animate_quadcopter_payload

plt.close('all')    # close any existing figures
anim_path = Path("sim/animations") / "quadcopter_payload.mp4"

# RK4 numerical integrator
def rk4_step(func, state, control, dt, params):
    k1 = func(state, control, params)
    k2 = func(state + 0.5 * dt * k1, control, params)
    k3 = func(state + 0.5 * dt * k2, control, params)
    k4 = func(state + dt * k3, control, params)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ----------------------------------------------------------------------
# Simulation main
def run_simulation(params, T, dt):
    # Initial state 
    state = np.array([
        0.0, #x_q
        5.0, #z_q
        0.0, #theta
        0.0, #x_q_dot
        0.0, #z_q_dot
        0.0, #theta_dot
        params["payload"]["rope_length"], #l
        0.0, #phi
        0.0, #l_dot
        0.0 #phi_dot
    ], dtype=float)

    # Parameters
    d   = float(params["quad"]["arm_length"])
    m_q = float(params["quad"]["mass"])
    m_p = float(params["payload"]["mass"])
    g   = float(params["environment"]["gravity"])

    # Rotor limits
    T_MAX_ROTOR = float(params.get("actuators", {}).get("max_thrust_per_rotor", 20.0))
    TAU_MAX     = float(params.get("actuators", {}).get("max_pitch_torque", 5.0))
    TAU_MIN    = -TAU_MAX
    T_MAX_TOTAL = 4.0 * T_MAX_ROTOR

    # Load controller, gains and output limits
    Kp_x, Ki_x, Kd_x = float(params["pid"]["x"]["Kp"]), float(params["pid"]["x"]["Ki"]), float(params["pid"]["x"]["Kd"])
    u_min_x, u_max_x = float(params["pid"]["x"]["u_min"]), float(params["pid"]["x"]["u_max"])
    x_controller = PIDController(Kp_x, Ki_x, Kd_x, u_min_x, u_max_x)

    Kp_z, Ki_z, Kd_z = float(params["pid"]["z"]["Kp"]), float(params["pid"]["z"]["Ki"]), float(params["pid"]["z"]["Kd"])
    u_min_z, u_max_z = float(params["pid"]["z"]["u_min"]), float(params["pid"]["z"]["u_max"])
    z_controller = PIDController(Kp_z, Ki_z, Kd_z, u_min_z, u_max_z)

    Kp_theta, Kd_theta = float(params["pid"]["theta"]["Kp"]), float(params["pid"]["theta"]["Kd"])
    theta_controller = PIDController(Kp_theta, 0.0, Kd_theta, TAU_MIN, TAU_MAX) 

    Kp_phi, Kd_phi = float(params["pid"]["phi"]["Kp"]), float(params["pid"]["phi"]["Kd"]) 
    u_min_phi, u_max_phi = float(params["pid"]["phi"]["u_min"]), float(params["pid"]["phi"]["u_max"])
    phi_controller = PIDController(Kp_phi, 0.0, Kd_phi, u_min_phi, u_max_phi) #no integral: 0 deg is equilibrium point & no cst lateral disturbance

    # Desired references
    x_ref, z_ref = 4.0, 5.0  # initial position
    phi_ref = 0.0            # desired payload angle (rad)

    # Storage arrays
    time = np.arange(0.0, T, dt)
    quad_traj, payload_traj = [], []

    # For plotting time histories
    x_hist, z_hist, theta_hist, phi_hist = [], [], [], []
    x_des_hist, z_des_hist, theta_des_hist, phi_des_hist = [], [], [], []

    # ----------------------------------------------------------------------
    for t in time:
        # Unpack states
        x_q, z_q, theta, xdot, zdot, thetadot, l, phi, ldot, phidot = state

        #add step changes in reference position at different times 
        # if t >= 2.0:
        #     x_ref = 4.0
        #     z_ref = 5.0
        #     #z_ref = 7.0
        # #elif t > 8:
        #  #   x_ref = 4.0
        #   #  z_ref = 5.0
    
        # --- Quadcopter Position Control ---
        a_x_pos = x_controller.update(x_ref - x_q, dt) 
        a_z_des = z_controller.update(z_ref - z_q, dt)

        # --- Payload Swing Control ---
        a_x_swing = phi_controller.update(phi_ref - phi, dt)

        # Total desired lateral acceleration
        a_x_des = a_x_pos + a_x_swing # sum contributions to lat. acceleration from position and swing controllers

        # Desired pitch from lateral accel
        theta_des = -np.arctan2(a_x_des, max(g, 1e-6)) 

        # Desired thrust magnitude (vertical control)
        T_total_des = m_q * (g + a_z_des) / max(np.cos(theta), 0.1)
        T_total_des = np.clip(T_total_des, 0.0, T_MAX_TOTAL)

        # Attitude control
        tau_des = theta_controller.update(theta_des - theta, dt)

        #theta_err = theta_des - theta
        #tau_des = Kp_theta * theta_err - Kd_theta * thetadot
        #tau_des = np.clip(tau_des, -TAU_MAX, TAU_MAX)

        # --- Mixer ---
        pair_front = 0.5 * T_total_des + 0.5 * tau_des / max(d, 1e-6)
        pair_rear  = 0.5 * T_total_des - 0.5 * tau_des / max(d, 1e-6)
        T1 = T2 = 0.5 * pair_front
        T3 = T4 = 0.5 * pair_rear

        T1 = np.clip(T1, 0.0, T_MAX_ROTOR)
        T2 = np.clip(T2, 0.0, T_MAX_ROTOR)
        T3 = np.clip(T3, 0.0, T_MAX_ROTOR)
        T4 = np.clip(T4, 0.0, T_MAX_ROTOR)

        # Recompute totals (for consistency)
        T_total = T1 + T2 + T3 + T4
        tau = d * ((T1 + T2) - (T3 + T4))

        # Rope feed (disabled)
        u_l = 0.0

        control = np.array([T1, T2, T3, T4, u_l], dtype=float)

        # Integrate dynamics
        state = rk4_step(derivatives, state, control, dt, params)

        # Record trajectories
        xq, zq, _, _, _, _, l, phi, _, _ = state
        u_vec = np.array([np.sin(phi), -np.cos(phi)])
        xp, zp = xq + l * u_vec[0], zq + l * u_vec[1]
        quad_traj.append([xq, zq])
        payload_traj.append([xp, zp])

        # Record states for time histories
        x_hist.append(x_q)
        z_hist.append(z_q)
        theta_hist.append(theta)
        phi_hist.append(phi)
        phi_des_hist.append(phi_ref)  # always zero
        x_des_hist.append(x_ref)
        z_des_hist.append(z_ref)
        theta_des_hist.append(theta_des)

    # Convert to arrays
    return (
        np.array(time),
        np.array(quad_traj),
        np.array(payload_traj),
        np.array(x_hist),
        np.array(z_hist),
        np.array(theta_hist),
        np.array(phi_hist),
        np.array(phi_des_hist),
        np.array(x_des_hist),
        np.array(z_des_hist),
        np.array(theta_des_hist),
    )

# ----------------------------------------------------------------------
if __name__ == "__main__":
    sim_dir = Path(__file__).parent
    candidates = [sim_dir / "config.yaml", sim_dir.parent / "config.yaml"]
    config_file = next((p for p in candidates if p.exists()), None)
    if config_file is None:
        raise FileNotFoundError(f"Config file not found. Searched: {candidates}")
    params = load_params(str(config_file))

    T, dt = 15.0, 0.01
    (
        time,
        quad_traj,
        payload_traj,
        x_hist,
        z_hist,
        theta_hist,
        phi_hist,
        phi_des_hist,
        x_des_hist,
        z_des_hist,
        theta_des_hist,
    ) = run_simulation(params, T, dt)

    # --- Animation --- uncomment these lines if you want to produce an mp4 animation
    animate_quadcopter_payload(time, quad_traj, payload_traj, theta_hist, params,
                           filename=anim_path.name, out_dir = anim_path.parent, fps=30)
    

    # Get output directory (same as script location)
    output_dir = Path(__file__).parent

    # ----------------------------------------------------------------------
    # PLOTTING
    # ----------------------------------------------------------------------

    # --- 1. Trajectory Plot ---
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.plot(quad_traj[:, 0], quad_traj[:, 1], label="Quadcopter")
    ax1.plot(payload_traj[:,0], payload_traj[:,1], label="Payload")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("z [m]")
    ax1.legend()
    ax1.set_title("Quadcopter + Payload Swing")
    ax1.grid()
    ax1.axis("equal")
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # --- 2. Time History Plots ---
    fig2, axs2 = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
    
    # z position
    axs2[0].plot(time, z_hist, label=r"$z_q$")
    axs2[0].plot(time, z_des_hist, '--', label=r"$z_{ref}$")
    axs2[0].set_ylabel("Altitude [m]")
    axs2[0].legend()
    axs2[0].grid(True)

    # x position
    axs2[1].plot(time, x_hist, label=r"$x_q$")
    axs2[1].plot(time, x_des_hist, '--', label=r"$x_{ref}$")
    axs2[1].set_ylabel("Position X [m]")
    axs2[1].legend()
    axs2[1].grid(True)

    # pitch
    axs2[2].plot(time, np.degrees(theta_hist), label=r"$\theta$ [deg]")
    axs2[2].plot(time, np.degrees(theta_des_hist), '--', label=r"$\theta_{des}$ [deg]")
    axs2[2].set_ylabel("Pitch [deg]")
    axs2[2].set_xlabel("Time [s]")
    axs2[2].legend()
    axs2[2].grid(True)

    # swing
    axs2[3].plot(time, np.degrees(phi_hist), label=r"$\phi$ [deg]")
    axs2[3].plot(time, np.degrees(phi_des_hist), '--', label=r"$\phi_{des}$ [deg]")
    axs2[3].set_ylabel("Swing, phi [deg]")
    axs2[3].set_xlabel("Time [s]")
    axs2[3].legend()
    axs2[3].grid(True)

    fig2.suptitle("Quadcopter Position and Attitude Tracking")
    plt.tight_layout()
    plt.savefig(output_dir / 'time_histories.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Remove or comment out all plt.show() calls
