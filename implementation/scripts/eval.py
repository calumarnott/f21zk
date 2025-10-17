import os, argparse, json
from pathlib import Path

from pendulum_ml.utils import parse_with_config
from pendulum_ml.evaluate import evaluate_test
from pendulum_ml.utils import load_cfg

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (e.g., models/checkpoints/<run>.pt)")
    p.add_argument("--run", default=None, help="Run id (experiments/<run>). If omitted, inferred from ckpt name.")

    # ---- NEW FLAGS for adversarial evaluation ----
    p.add_argument("--attack", default=None, help="Attack method: pgd, fgsm, spsa, or none")
    p.add_argument("--eps", type=float, default=0.03, help="Attack epsilon (perturbation budget)")
    p.add_argument("--steps", type=int, default=10, help="Number of attack iterations (for PGD/SPSA)")
    p.add_argument("--alpha", type=float, default=None, help="Attack step size (optional)")
    p.add_argument("--norm", default="linf", help="Attack norm: linf or l2")

    args = p.parse_args()

    # Infer run id if not given: use the ckpt basename without extension
    inferred_run = os.path.splitext(os.path.basename(args.ckpt))[0]
    run = args.run or inferred_run

    path = Path("experiments") / run / "config.json"
    if not path.exists():
        raise SystemExit(f"Config not found at {path}. Provide a valid --run.")
    cfg = load_cfg(str(path)) 

    # ---- Optional attack config ----
    attack_cfg = None
    if args.attack is not None and args.attack.lower() != "none":
        attack_cfg = {
            "method": args.attack,
            "eps": args.eps,
            "steps": args.steps,
            "alpha": args.alpha,
            "norm": args.norm,
        }
        print(f"[INFO] Evaluating under adversarial attack: {attack_cfg['method']} (eps={attack_cfg['eps']}, steps={attack_cfg['steps']})")


    # ---- Evaluate (clean or adversarial) ----
    results = evaluate_test(cfg, args.ckpt, run=run, loaders=None, show_progress=True, attack_cfg=attack_cfg)
    print(json.dumps(results, indent=2))