import argparse
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt

if __name__ == "__main__":
    
    p=argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run id (experiments/<run>)")
    
    args = p.parse_args()
    
    # create figures directory if it doesn't exist
    path = Path("experiments") / args.run
    figs = path / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    
    # load metrics CSV
    df = pd.read_csv(path/"metrics.csv")
    
    # plot training and validation loss
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train MSE")
    plt.plot(df["epoch"], df["val_loss"], label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(figs/"loss.png", dpi=200)
