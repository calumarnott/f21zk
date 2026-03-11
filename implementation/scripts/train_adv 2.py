from pendulum_ml.utils import parse_with_config
from pendulum_ml.train_adv import train_adv

if __name__ == "__main__":
    cfg, _ = parse_with_config()
    cfg["train"]["adv"]["enabled"] = True
    train_adv(cfg)
