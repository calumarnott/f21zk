# scripts/csv_to_idx.py
import argparse, struct, numpy as np
from pathlib import Path
import json, torch
from pendulum_ml.models.registry import make_model
from pendulum_ml.utils import import_system

# --- minimal IDX writer for float32 ---
def write_idx_float32(path: Path, arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float32)
    dims = arr.shape
    with path.open("wb") as f:
        # magic: 00 00 type dims
        f.write(bytes([0, 0, 0x0D, len(dims)]))  # 0x0D = float32
        for d in dims:
            f.write(struct.pack(">I", int(d)))   # big-endian dimension sizes
        f.write(arr.tobytes(order="C"))

def load_model(ckpt: Path):
    run = ckpt.stem
    cfg = json.loads((Path("experiments")/run/"config.json").read_text())
    device = torch.device(cfg.get("device","cpu"))
    model_name = cfg["model"]["name"]
    kwargs = cfg["model"].get(model_name, {})
    model = make_model(model_name, **kwargs).to(device)
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd, strict=False)
    model.eval()
    cps = import_system(cfg["system"])
    return model, cfg, cps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)         
    args = ap.parse_args()


    outdir = Path("data/processed/pendulum")
    outdir.mkdir(parents=True, exist_ok=True)
    path_csv = Path("data/raw/pendulum/traj_005.csv")
    # load CSV: expects header theta,theta_dot
    data = np.genfromtxt(path_csv, delimiter=",", names=True)
    X2 = np.stack([data["theta"], data["theta_dot"]], axis=1).astype(np.float32)  # [N,3]


    # compute u_ref offline
    model, cfg, cps = load_model(Path(args.ckpt))
    # Build the model input the same way your VCL does: [theta, theta_dot, error],
    # where error = theta - setpoint (in problem space). Normalisation is inside the net if needed; 
    # if you normalise outside, mimic exactly what your VCL 'normalise' does here.
    setpoint = np.float32(0.0)
    
    X3 = np.zeros((X2.shape[0], 3), dtype=np.float32)  # [N,3]
    with torch.no_grad():
        u_ref = model(torch.from_numpy(X3).float()).cpu().numpy()  # [N,1] (assumed)

    # write IDX files
    write_idx_float32(outdir/"pend_states.idx", X2)     # [N,3]
    write_idx_float32(outdir/"u_ref.idx", u_ref)        # [N,1]
    print("Wrote:", outdir/"pend_states.idx", outdir/"u_ref.idx")

if __name__ == "__main__":
    main()
