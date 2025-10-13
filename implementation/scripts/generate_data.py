from pendulum_ml.utils import parse_with_config, import_system
from pendulum_ml.data.generate import simulate
from pendulum_ml.data.process import to_processed
from pathlib import Path
import sys, yaml


if __name__ == "__main__":
    """ Generate and process data based on configuration. """
    
    if "--animate" in sys.argv:
        import sys
        sys.argv.remove("--animate") # remove it so parse_with_config doesn't get confused
        animate_flag = True
    else:
        animate_flag = False
        
    cfg, _ = parse_with_config() # get config from command-line args
    
    
    # simulate and save raw data
    raw_path = simulate(cfg, out_dir=f"data/raw")
    
    cps = import_system(cfg["system"])
    
    if animate_flag:
        animation_path = cps.animate(cfg, trajectory_path=raw_path, fps=30)
        print(f"Animation saved to: {animation_path}")
        
    # save config used for this data to data/raw/<system>/configs/traj_name_from_raw_path.yaml
    config_dir = Path("data/raw") / cfg["system"] / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / (Path(raw_path).stem + ".yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"Config used for this data saved to: {config_path}")
    
    # process raw data and save processed tensors
    to_processed(raw_path, cfg, out_dir=f"data/processed/{cfg['system']}")
