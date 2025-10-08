import numpy as np
import yaml
from pathlib import Path

# Load parameters from YAML
def load_params(config_path="config.yaml"):
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file.resolve()}")
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

# Unit vectors based on rope swing angle
def rope_vectors(phi):
    """Return unit vector along rope (quad -> payload) and tangential vector."""
    u = np.array([np.sin(phi), -np.cos(phi)])  # along rope
    t = np.array([np.cos(phi), np.sin(phi)])  # tangential, direction of phi+
    return u, t

# Dynamics function
def derivatives(state, control, params):
    """
    Compute state derivatives for quad + rope payload system in 2D.

    State vector:
      [xq, zq, theta, xq_dot, zq_dot, theta_dot, l, phi, l_dot, phi_dot]
    Control vector:
      [T1, T2, T3, T4, u_l]
    """
    # Unpack state
    xq, zq, theta, xq_dot, zq_dot, theta_dot, l, phi, l_dot, phi_dot = state
    T1, T2, T3, T4, u_l = control

    # Params
    m_q = params["quad"]["mass"]
    I_q = params["quad"]["inertia"]
    d = params["quad"]["arm_length"]

    m_p = params["payload"]["mass"]

    g = params["environment"]["gravity"]

    winch_model = params["winch"]["model"]
    omega_l = params["winch"].get("omega", 0.0)

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