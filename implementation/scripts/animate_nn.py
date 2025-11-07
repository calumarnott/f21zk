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
    p.add_argument("--not-animate", action="store_true", help="Generate animation video.")
    p.add_argument("--plot", action="store_true", help="Also generate plots of state variables over time.")
    
    args = p.parse_args()

    plot_graphs = args.plot # boolean flag
    animate = not args.not_animate # boolean flag

    inferred_run = os.path.splitext(os.path.basename(args.ckpt))[0]
    run = args.run or inferred_run
    
    path = Path("experiments") / run / "config.json"
    if not path.exists():
        raise SystemExit(f"Config not found at {path}. Provide a valid --run.")
    cfg = load_cfg(str(path))
    cps = import_system(cfg["system"])
    
    out_dir = Path("experiments") / run / "animations" / f"{Path(args.ckpt).stem}"
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
    
    kwargs = cfg["model"].get(cfg["model"]["name"], {})
    model = make_model(
        cfg["model"]["name"],
        **kwargs
    ).to(device)
    
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    trajectory = generate_trajectory_from_NN_controller(cfg, cps, model, initial_states)
    
    out_path = cps.animate(cfg, trajectory, out_path=out_path, plot=plot_graphs, animate=animate)
    print(f"Animation saved to {out_path}")
    