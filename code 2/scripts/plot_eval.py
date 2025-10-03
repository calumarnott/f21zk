import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    
    # Parse command-line args
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="<run> id (in experiments/<run>)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = p.parse_args()

    # Create figures directory if it doesn't exist
    run_dir = Path("experiments") / args.run
    figs = run_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # Load predictions CSV
    preds_path = run_dir / f"predictions_{args.split}.csv"
    
    if not preds_path.exists():
        raise SystemExit(f"Missing {preds_path}. Run evaluation with predictions first.")

    # Load predictions and true values
    df = pd.read_csv(preds_path)
    y, yhat = df["y_true"].values, df["y_pred"].values
    lo, hi = min(y.min(), yhat.min()), max(y.max(), yhat.max())

    # Parity plot
    plt.figure()
    plt.scatter(y, yhat, s=10, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Parity plot ({args.split})")
    plt.tight_layout()
    plt.savefig(figs / f"parity_{args.split}.png", dpi=200)
    plt.close()

    # Draw test MSE as a dashed line on the loss curve (if test)
    if args.split == "test":
        summary_path = run_dir / "metrics.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            test_mse = summary.get("test_mse")
            if test_mse is not None:
                # Append test line onto existing loss plot if it exists
                loss_plot = figs / "loss.png"
                # If you want to regenerate loss.png with a dashed line, do that in your plot_metrics.py
                print(f"test_mse={test_mse}")
