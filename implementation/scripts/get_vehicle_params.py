import json, argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--norms", required=True, help=".../data/processed/pendulum/norms.json")
p.add_argument("--out",   required=True, help=".../specs/pendulum_params.cli")
p.add_argument("--input-dim", type=int, default=2)  # 2 or 3
p.add_argument("--epsilon", type=float, required=True)  # from config.verify.eps
p.add_argument("--setpoint", type=float, default=0.0)
args = p.parse_args()

stats = json.loads(Path(args.norms).read_text())
means = stats["mu"][0]
stds  = stats["sd"][0]

lines = [
    f"--parameter mu_theta:{means[0]}",
    f"--parameter mu_theta_dot:{means[1]}",
    f"--parameter sd_theta:{stds[0]}",
    f"--parameter sd_theta_dot:{stds[1]}",
    f"--parameter epsilon:{args.epsilon}",
]

if args.input_dim == 3:
    lines += [
        f"--parameter mu_err:{means[2]}",
        f"--parameter sd_err:{stds[2]}",
        f"--parameter setpoint:{args.setpoint}",
    ]

Path(args.out).write_text("\n".join(lines) + "\n")
print(f"Wrote {args.out}")
