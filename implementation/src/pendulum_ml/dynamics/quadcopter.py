import numpy as np
from dataclasses import dataclass

STATE_NAMES = ["x", "z", "theta", "xq_dot", "zq_dot",\
                "theta_dot", "l", "phi", "l_dot", "phi_dot"]
CONTROL_AXES = ["x", "z", "theta", "phi", "l"]


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
    u_dict["l"] = 0.0  # no winch control for now

    # Integrate dynamics
    state = step(state, u_dict, f, params, dt)
    
    err_dict["x"] = float(x_ref - x_q)
    err_dict["z"] = float(z_ref - z_q)
    err_dict["theta"] = float(theta_des - theta)
    err_dict["phi"] = float(phi_ref - phi)
    err_dict["l"] = 0.0


    return state, u_dict, err_dict

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
    u_l = control["l"]

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