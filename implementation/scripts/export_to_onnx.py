import torch, argparse, json
from pathlib import Path
from pendulum_ml.models.registry import make_model
from pendulum_ml.utils import import_system


def load_snapshot(run: str):
    snap = Path("experiments")/run/"config.json"
    if not snap.exists():
        raise SystemExit(f"Missing snapshot: {snap}")
    return json.loads(snap.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="models/onnx/pendulum_model.onnx")
    args = ap.parse_args()
    
    
    run = Path(args.ckpt).stem
    cfg = load_snapshot(run)
    device = cfg.get("device", "cpu")
    device = torch.device(device)
    
    model_name = cfg["model"]["name"]

    kwargs = cfg["model"].get(model_name, {})
    model = make_model(model_name, **kwargs).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd, strict=False)
    model.eval()

    dummy = torch.zeros(1, 3, dtype=torch.float32)
    out_dim = int(cfg["model"].get("out_dim", 1))
    
    cps = import_system(cfg["system"])
    input_names = cps.STATE_NAMES + [f"error_{axis}" for axis in cps.INPUT_ERROR_AXES]
    output_names = [f"u_{axis}" for axis in cps.OUTPUT_CONTROL_AXES]
    
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["output"],
        opset_version=13
    )
    print(f"Exported ONNX -> {args.out}")

