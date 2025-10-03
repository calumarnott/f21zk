from pendulum_ml.utils import parse_with_config
from pendulum_ml.train import train

if __name__ == "__main__":
    cfg, _ = parse_with_config()
    train(cfg)
