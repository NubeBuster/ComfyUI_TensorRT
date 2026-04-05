#!/usr/bin/env python3
"""Quick ONNX export + weight map generation. No engine build.

Usage: startcomfy exec python /app/custom_nodes/comfyui_tensorrt/tests/gen_weight_map.py \
         <checkpoint_path> [--output /path/to/output.weight_map.json]
"""
import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, "/app")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("--output", required=True, help="Output .weight_map.json path")
    parser.add_argument("--model-type", default="sdxl")
    args = parser.parse_args()

    import comfy.sd
    import comfy.model_management

    # Load checkpoint
    log.info("Loading checkpoint: %s", args.checkpoint_path)
    result = comfy.sd.load_checkpoint_guess_config(
        args.checkpoint_path, output_vae=False, output_clip=False,
        output_clipvision=False, output_model=True
    )
    model_patcher = result[0]
    comfy.model_management.load_models_gpu([model_patcher])
    unet = model_patcher.model.diffusion_model
    unet_config = model_patcher.model.model_config.unet_config
    dtype = torch.float16

    in_channels = unet_config.get("in_channels", 4)
    context_dim = unet_config.get("context_dim", None)
    if isinstance(context_dim, list):
        context_dim = context_dim[0]
    y_dim = unet_config.get("adm_in_channels", None) or unet_config.get("vec_in_dim", None)

    # ONNX export wrapper (same as tensorrt_convert.py)
    class UNET(torch.nn.Module):
        def forward(self, x, timesteps, context, *extra_args):
            extras = {}
            for i, name in enumerate(self.extra_input_names):
                if i < len(extra_args):
                    extras[name] = extra_args[i]
            return self.unet(x, timesteps, context,
                             transformer_options=self.transformer_options, **extras)

    wrapper = UNET()
    wrapper.unet = unet
    wrapper.transformer_options = model_patcher.model_options.get("transformer_options", {}).copy()
    input_names = ["x", "timesteps", "context"]
    extra_input_names = []
    if y_dim:
        input_names.append("y")
        extra_input_names.append("y")
    wrapper.extra_input_names = extra_input_names

    batch, h, w = 1, 128, 128  # 1024x1024 / 8
    dummy = [
        torch.randn(batch, in_channels, h, w, dtype=dtype, device="cuda"),
        torch.zeros(batch, dtype=dtype, device="cuda"),
        torch.randn(batch, 77, context_dim, dtype=dtype, device="cuda"),
    ]
    if y_dim:
        dummy.append(torch.randn(batch, y_dim, dtype=dtype, device="cuda"))

    onnx_dir = tempfile.mkdtemp(prefix="trt_map_")
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    log.info("Exporting ONNX...")
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper, tuple(dummy), onnx_path,
            input_names=input_names, output_names=["h"],
            opset_version=17, dynamo=False,
        )
    log.info("ONNX exported: %s", onnx_path)

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    torch.cuda.empty_cache()

    # Build weight map
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    weight_map = {}
    for node in model.graph.node:
        if not node.name:
            continue
        for inp in node.input:
            if not inp.startswith("onnx::"):
                continue
            parts = node.name.strip("/").split("/")
            if parts and parts[0] == "unet":
                parts = parts[1:]
            op_suffix = parts[-1] if parts else ""
            path_parts = parts[:-1]
            if not path_parts:
                continue
            pytorch_key = ".".join(path_parts)
            if "MatMul" in op_suffix:
                pytorch_key += ".weight"
            elif "Add" in op_suffix:
                pytorch_key += ".bias"
            else:
                pytorch_key += ".weight"
            # Fix doubled path segments from nn.Sequential containers
            # Only dedup non-numeric segments (numeric = ModuleList indices)
            pytorch_key = re.sub(r"\.([a-zA-Z_]\w*)\.\1\.", r".\1.", pytorch_key)
            weight_map[inp] = pytorch_key
    del model

    log.info("Weight map: %d entries", len(weight_map))
    matmul_count = sum(1 for k in weight_map if "MatMul" in k)
    log.info("  MatMul entries: %d", matmul_count)

    # Show samples
    for k, v in list(weight_map.items())[:5]:
        log.info("  %s -> %s", k, v)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(weight_map, f)
    log.info("Saved: %s", args.output)

    # Cleanup
    shutil.rmtree(onnx_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
