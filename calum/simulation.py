import numpy as np
import matplotlib.pyplot as plt
from dynamics import load_params, derivatives

# RK4 numerical integrator to deal with the ODEs
def rk4_step(func, state, control, dt, params): #func is the derivatives() from dynamics.py
    k1 = func(state, control, params)
    k2 = func(state + 0.5 * dt * k1, control, params)
    k3 = func(state + 0.5 * dt * k2, control, params)
    k4 = func(state + dt * k3, control, params)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Simulation main
def run_simulation(params, T, dt):
    # Initial state
    # [xq, zq, theta, xq_dot, zq_dot, theta_dot, l, phi, l_dot, phi_dot]
    state = np.array([
        0.0,     # x_q
        0.0,     # z_q
        0.0,     # theta
        0.0,     # x_q_dot
        0.0,     # z_q_dot
        0.0,     # theta_dot
        params["payload"]["rope_length"],  # l
        0.2,     # phi (rad) initial swing angle
        0.0,     # l_dot
        0.0      # phi_dot
    ])

    m_q = params["quad"]["mass"]
    m_p = params["payload"]["mass"]
    g   = params["environment"]["gravity"]

    T_hover = (m_q + m_p) * g / 4.0   # Thrust hover per rotor (we can change this to simulate climb etc)
    u_l = 1.0 # rope feed rate m/s
    control = np.array([T_hover, T_hover, T_hover, T_hover, u_l]) #control output vector

    # Storage for plotting
    time = np.arange(0, T, dt)
    # store coords of quad and payload at each step
    quad_traj = []
    payload_traj = []

    for t in time:
        # Integrate
        state = rk4_step(derivatives, state, control, dt, params)

        # Save trajectories
        xq, zq, _, _, _, _, l, phi, _, _ = state #extract relevant states for traj
        # Payload position = quad position + l * u(phi)
        u = np.array([np.sin(phi), -np.cos(phi)])
        xp, zp = xq + l*u[0], zq + l*u[1]

        quad_traj.append([xq, zq])
        payload_traj.append([xp, zp])

    return np.array(time), np.array(quad_traj), np.array(payload_traj)

# Entry point
if __name__ == "__main__":
    params = load_params("config.yaml")

    T = 15.0 #s
    dt = 0.01 #s
    time, quad_traj, payload_traj = run_simulation(params, T, dt)

    # Plot results
    plt.figure(figsize=(8,6))
    plt.plot(quad_traj[:,0], quad_traj[:,1], label="Quadcopter")
    plt.plot(payload_traj[:,0], payload_traj[:,1], label="Payload")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.legend()
    plt.title("Quadcopter + Payload Swing (uncontrolled)")
    plt.grid()
    plt.axis("equal")
    plt.show()
