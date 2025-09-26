import argparse
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt

if __name__ == "__main__":
    
    p=argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run id (experiments/<run>)")
    
    a=p.parse_args()
    
    # create figures directory if it doesn't exist
    run=Path(a.run)
    figs=run/"figures"
    figs.mkdir(parents=True, exist_ok=True)
    
    # load metrics CSV
    df = pd.read_csv(run/"metrics.csv")
    
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
