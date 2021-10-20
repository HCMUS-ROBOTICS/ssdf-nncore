import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.onnx
import yaml

from nncore.utils import getter

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export model to onnx")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument('--in_shape', type=int, nargs="+", required=True)
    parser.add_argument('--inputs', type=str, nargs='+', required=True)
    parser.add_argument('--outputs', type=str, nargs='+', required=True)
    args = parser.parse_args()

    # Config from checkpoint
    ckpt_dir: Path = args.checkpoint.parent
    with (ckpt_dir / "config.yaml").open() as config_file:
        config = yaml.safe_load(config_file)

    # Load model
    model_config = config["pipeline"]["model"]
    model: nn.Module = getter.get_instance(model_config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    model_name: str = model_config["name"]
    in_shape_str = "x".join(map(str, args.in_shape))
    onnx_name = f"{model_config['name']}-{in_shape_str}.onnx"
    torch.onnx.export(model, torch.randn(args.in_shape), onnx_name, opset_version=args.opset,
                      input_names=args.inputs, output_names=args.outputs)

    print(f"Exported to {onnx_name}")
