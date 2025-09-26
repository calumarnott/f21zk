from src.pendulum_ml.utils import parse_with_config
from src.pendulum_ml.data.generate import simulate
from src.pendulum_ml.data.process import to_processed

if __name__ == "__main__":
    cfg, _ = parse_with_config()
    raw_paths = simulate(cfg, out_dir=f"data/raw/{cfg['system']}")
    to_processed(raw_paths, cfg, out_dir=f"data/processed/{cfg['system']}")
