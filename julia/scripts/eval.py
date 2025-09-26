import os, argparse, json
from pathlib import Path

from pendulum_ml.utils import parse_with_config
from pendulum_ml.evaluate import evaluate_test
from pendulum_ml.utils import load_cfg

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (e.g., models/checkpoints/<run>.pt)")
    p.add_argument("--run", default=None, help="Run id (experiments/<run>). If omitted, inferred from ckpt name.")
    args = p.parse_args()

    # Infer run id if not given: use the ckpt basename without extension
    inferred_run = os.path.splitext(os.path.basename(args.ckpt))[0]
    run = args.run or inferred_run

    snap = Path("experiments") / run / "config.json"
    if not snap.exists():
        raise SystemExit(f"Config not found at {snap}. Provide a valid --run.")
    cfg = load_cfg(str(snap)) 

    results = evaluate_test(cfg, args.ckpt, run=run, loaders=None, show_progress=True)
    
    print(results)
