import os, argparse
from pathlib import Path
import numpy as np
from pendulum_ml.utils import import_system, load_cfg, apply_overrides
from pendulum_ml.animate import generate_trajectory_from_NN_controller
import torch
from pendulum_ml.models.registry import make_model

if __name__ == "__main__":
    """ Animate a trained neural network model on test data. """
    
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (e.g., models/checkpoints/<run>.pt)")
    p.add_argument("--run", default=None, help="Run id (experiments/<run>). If omitted, inferred from ckpt name.")
    
    args = p.parse_args()
    
    inferred_run = os.path.splitext(os.path.basename(args.ckpt))[0]
    run = args.run or inferred_run
    
    path = Path("experiments") / run / "config.json"
    if not path.exists():
        raise SystemExit(f"Config not found at {path}. Provide a valid --run.")
    cfg = load_cfg(str(path))
    cps = import_system(cfg["system"])
    
    out_dir = Path("experiments") / run / "animations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(args.ckpt).stem + ".mp4")
    
    initial_states_cfg_path = Path("data") / "raw" / cfg["system"] / "configs"
    if not initial_states_cfg_path.exists():
        raise SystemExit(f"Initial states config not found at {initial_states_cfg_path}. Ensure you have generated data for this system.")
    
    # the cfg has the name traj_XXX.csv, load the last one
    traj_cfgs = sorted(initial_states_cfg_path.glob("traj_*.yaml"))
    if not traj_cfgs:
        raise SystemExit(f"No traj_*.yaml files found in {initial_states_cfg_path}. Ensure you have generated data for this system.")
    initial_states_cfg = load_cfg(str(traj_cfgs[-1]))
    initial_states = np.array(initial_states_cfg["data"]["initial_state"])
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model = make_model(
        cfg["model"]["name"],
        in_dim=int(cfg["model"]["in_dim"]),
        out_dim=int(cfg["model"]["out_dim"]),
        hidden=tuple(cfg["model"]["hidden"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    trajectory = generate_trajectory_from_NN_controller(cfg, cps, model, initial_states)
    
    
    #################################
    ### Create plot for quadrotor ###
    #################################

    # Make three plots: x vs time, z vs time and phi vs time for each of the trajectories
    # Make one image with three subplots for each trajectory
    # Make plot in blue and the setpoint of each (x=4, z=5, phi=0) in orange dashed line
    import matplotlib.pyplot as plt
    num_trajectories = initial_states.shape[0]
    time = trajectory[:, 0].reshape(num_trajectories, -1)
    x = trajectory[:, 1].reshape(num_trajectories, -1)
    z = trajectory[:, 2].reshape(num_trajectories, -1)
    phi = trajectory[:, 8].reshape(num_trajectories, -1)
    
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
        axs[2].plot(time[i], phi[i], label='phi (roll angle)', color='blue')
        axs[2].axhline(y=0.0, color='orange', linestyle='--', label='phi setpoint')
        axs[2].set_title('phi vs Time')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('phi (rad)')
        axs[2].legend()
        axs[2].grid()
        
        plt.tight_layout()
        plot_path = out_dir / f"trajectory_{i+1}_plots.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Plots saved to {plot_path}")


    ########################################

    out_path = cps.animate(cfg, trajectory, out_path=out_path)
    print(f"Animation saved to {out_path}")
    
    