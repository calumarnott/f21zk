import argparse
from src.pendulum_ml.utils import convert_to_onnx

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch model (.pt/.pth) to ONNX format."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.json used for training"
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the saved model checkpoint (.pt or .pth)"
    )
    parser.add_argument(
        "--onnx",
        default=None,
        help="Optional output path for the ONNX file (defaults to same as ckpt)"
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

    # If user provides a flat list (e.g. 15), make it a tuple
    input_shape = tuple(args.input_shape) if args.input_shape else None

    convert_to_onnx(
        config_path=args.config,
        ckpt_path=args.ckpt,
        onnx_path=args.onnx,
        input_shape=input_shape,
        opset_version=args.opset,
        device=args.device
    )

if __name__ == "__main__":
    main()
