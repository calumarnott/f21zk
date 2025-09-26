import argparse
import os

from pendulum_ml.utils import parse_with_config
from pendulum_ml.evaluate import evaluate_test

if __name__ == "__main__":
    cfg, _ = parse_with_config()

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ckpt", required=True, help="Path to models/checkpoints/<run>.pt")
    ap.add_argument("--run", default=None, help="Experiment run id (experiments/<run>)")
    args, _ = ap.parse_known_args()

    # If --run omitted, infer from ckpt file name
    run = args.run or os.path.splitext(os.path.basename(args.ckpt))[0]

    results = evaluate_test(cfg, args.ckpt, run=run, loaders=None, show_progress=True)
    
    print(results)
