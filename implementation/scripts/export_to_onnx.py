import torch, argparse, json
from pathlib import Path
from pendulum_ml.models.registry import make_model


def load_snapshot(run: str):
    snap = Path("experiments")/run/"config.json"
    if not snap.exists():
        raise SystemExit(f"Missing snapshot: {snap}")
    return json.loads(snap.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--input-dim", type=int, required=True)   # 2 or 3
    ap.add_argument("--output-dim", type=int, default=1)
    args = ap.parse_args()
    
    
    run = Path(args.ckpt).stem
    cfg = load_snapshot(run)
    device = cfg.get("device", "cpu")
    device = torch.device(device)
    
    kwargs = cfg["model"].get(cfg["model"]["name"], {})
    model = make_model(args.arch, **kwargs).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd, strict=False)
    model.eval()

    dummy = torch.zeros(1, args.input-dim if hasattr(args,'input-dim') else args.input_dim)
    # some Python versions don't like hyphenated attribute; fallback:
    try:
        inpdim = args.input_dim
    except:
        inpdim = getattr(args, "input-dim")
    dummy = torch.zeros(1, inpdim, dtype=torch.float32)

    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["output"],
        opset_version=13
    )
    print(f"Exported ONNX -> {args.out} (input-dim={inpdim})")

if __name__ == "__main__":
    main()
