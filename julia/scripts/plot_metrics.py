import argparse
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--run", required=True)
    a=p.parse_args()
    run=Path(a.run); figs=run/"figures"; figs.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(run/"metrics.csv")
    plt.figure(); plt.plot(df["epoch"], df["train"], label="Train MSE"); plt.plot(df["epoch"], df["val"], label="Val MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Training and Validation Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(figs/"loss.png", dpi=200)
