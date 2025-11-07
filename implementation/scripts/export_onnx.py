import argparse
from pathlib import Path
import sys

# Add parent/src to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from pendulum_ml.utils import convert_to_onnx

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch model (.pt/.pth) to ONNX format."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name (e.g. clean_model-20251026-214014). Paths will be derived automatically."
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for the project (default: current directory)"
    )
    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        default=None,
        help="Input shape excluding batch dim (e.g. 15 or 3 224 224)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to load model on (default: cpu)"
    )

    args = parser.parse_args()

    # Construct paths from model name
    base = Path(args.base_dir)
    model_name = args.model_name
    
    config_path = base / "experiments" / model_name / "config.json"
    ckpt_path = base / "models" / "checkpoints" / f"{model_name}.pt"
    onnx_path = base / "exports" / f"{model_name}.onnx"

    # Validate that required files exist
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Ensure export directory exists
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # If user provides a flat list (e.g. 15), make it a tuple
    input_shape = tuple(args.input_shape) if args.input_shape else None

    print(f"Config:      {config_path}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"ONNX output: {onnx_path}")
    
    convert_to_onnx(
        config_path=str(config_path),
        ckpt_path=str(ckpt_path),
        onnx_path=str(onnx_path),
        input_shape=input_shape,
        opset_version=args.opset,
        device=args.device
    )
    
    print(f"✓ Export complete: {onnx_path}")

if __name__ == "__main__":
    main()
