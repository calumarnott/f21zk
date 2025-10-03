import subprocess
import sys
import argparse
from pathlib import Path

# Resolve repo root once so we can run everything from a stable CWD
ROOT = Path(__file__).resolve().parents[1]


def sh(*args: str) -> None:
    """ Run a command as a subprocess, printing it first.
    
    Args:
        *args (str): Command and its arguments.
    """
    # For display, print exactly what the caller passed
    print("+", " ".join(map(str, args)))

    # Build the actual command (script path resolved against ROOT if relative)
    script = Path(args[0])
    script_abs = script if script.is_absolute() else (ROOT / script)
    cmd = [sys.executable, str(script_abs), *map(str, args[1:])]

    # Run from the repo root so all scripts see a consistent working directory
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    """ End-to-end test script that runs
      1) Generate data
      2) Train model
      3) Evaluate model
      4) Plot evaluation results
      5) Plot training metrics

    Saves run as "test_end2end" in experiments directory.
    
    Command-line arguments:
        --exp (str): Experiment name prefix for the end-to-end test. Default is "test_end2end".
    
    Raises:
        SystemExit: If no runs are found with the specified experiment name.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="test_end2end", help="Experiment name for the end-to-end test.")
    args = p.parse_args()

    # Step 1: Generate data
    sh("scripts/generate_data.py")

    # Step 2: Train model
    sh("scripts/train.py", "--exp", args.exp)

    # Step 3: Evaluate model (infer latest run with this prefix)
    experiments_dir = ROOT / "experiments"
    runs = sorted([d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith(args.exp)])
    if not runs:
        raise SystemExit(f"No runs found with name starting with {args.exp} in {experiments_dir}.")
    latest_run = runs[-1].name
    ckpt_path = ROOT / "models" / "checkpoints" / f"{latest_run}.pt"
    sh("scripts/eval.py", "--ckpt", str(ckpt_path), "--run", latest_run)

    # Step 4: Plot evaluation results
    sh("scripts/plot_eval.py", "--run", latest_run)

    # Step 5: Plot training metrics
    sh("scripts/plot_metrics.py", "--run", latest_run)
    
    
    
    