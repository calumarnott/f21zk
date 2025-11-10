import json, argparse
from pathlib import Path



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="pendulum", help="pendulum")
    # Optional args.out: defaults to .../specs/pendulum_params.cli
    p.add_argument("--out", default="specs/pendulum_params.cli")
    # Optional args.epsilon: defaults to 0.05
    p.add_argument("--epsilon", type=float, default=0.05)

    args = p.parse_args()

    norms = Path(f"data/processed/{args.model_name}/norms.json")

    stats = json.loads(norms.read_text())
    means = stats["mu"][0]
    stds  = stats["sd"][0]
    

    lines = [
        f"--parameter mu_theta:{means[0]}",
        f"--parameter mu_theta_dot:{means[1]}",
        f"--parameter mu_err:{means[2]}",
        f"--parameter sd_theta:{stds[0]}",
        f"--parameter sd_theta_dot:{stds[1]}",
        f"--parameter sd_err:{stds[2]}",
        f"--parameter epsilon:{args.epsilon}",
        f"--parameter setpoint:{0.0}"
    ]

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")
