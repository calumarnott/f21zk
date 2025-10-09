import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import numpy as np


def animate_quadcopter_payload(time, quad_traj, payload_traj, theta_hist, params,
                               filename="quadcopter_payload.mp4",
                               fps=30, out_dir="."):
    """
    Animate a 2D quadcopter + payload system using the simulated pitch angle.

    Parameters
    ----------
    time : array
        Time vector from simulation.
    quad_traj : Nx2 array
        Quadcopter (x, z) positions.
    payload_traj : Nx2 array
        Payload (x, z) positions.
    theta_hist : array
        Quadcopter pitch angle history (radians).
    params : dict
        System parameters loaded from config.
    filename : str
        Output filename (.mp4 or .gif).
    fps : int
        Frames per second for animation.
    out_dir : str or Path
        Directory to save the animation.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    d = params["quad"]["arm_length"]

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
        xp, zp = payload_traj[i]
        theta = theta_hist[i]

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

    # --- Animation ---
    anim = animation.FuncAnimation(
        fig, update, frames=len(time),
        init_func=init, blit=True, interval=1000 / fps
    )

    # Save or display
    print(f'Saving animation to {out_path} ...')
    if out_path.suffix.lower() == ".mp4":
        anim.save(out_path, writer="ffmpeg", fps=fps)
        print(f'Animation saved to {out_path}')
    elif out_path.suffix.lower() == ".gif":
        anim.save(out_path, writer="pillow", fps=fps)
        print(f'Animation saved to {out_path}')
    else:
        print(f"Unsupported extension: {out_path.suffix}, showing instead.")
        plt.show()

    plt.close(fig)
    return anim
