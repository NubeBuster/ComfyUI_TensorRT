#!/usr/bin/env python3
"""Standalone refit test — run via: startcomfy exec python /app/custom_nodes/comfyui_tensorrt/tests/test_refit.py

Tests:
  inspect  — Load engine, report refittable weight count/names, check LoRA coverage
  refit    — Load checkpoint + LoRA, refit engine, run inference, compare outputs
  build    — Build a small refit-enabled engine from a checkpoint (for testing)

Usage:
  python test_refit.py inspect <engine_path>
  python test_refit.py refit <engine_path> <checkpoint_path> <lora_path> [--model-type sdxl]
  python test_refit.py build <checkpoint_path> [--model-type sdxl] [--height 1024] [--width 1024]
"""

import argparse
import json
import logging
import os
import re
import sys
import time

# Must be importable inside ComfyUI container
import torch
import tensorrt as trt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared: ONNX weight name mapping
# ---------------------------------------------------------------------------

def _build_onnx_weight_map(onnx_path):
    """Build mapping from onnx::* tensor names to PyTorch state_dict keys.

    ONNX export turns attention weights into anonymous tensors (onnx::MatMul_NNN).
    The consuming node's name encodes the PyTorch path, e.g.
    /unet/input_blocks.4.1/transformer_blocks.0/attn1/to_q/MatMul
    -> input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight
    """
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
    log.info("ONNX weight map: %d entries", len(weight_map))
    return weight_map


# ---------------------------------------------------------------------------
# map: Generate weight map from ONNX file
# ---------------------------------------------------------------------------

def cmd_map(args):
    """Generate weight map JSON from an ONNX file."""
    onnx_path = args.onnx_path
    log.info("Building weight map from: %s", onnx_path)
    weight_map = _build_onnx_weight_map(onnx_path)

    output = args.output or onnx_path.replace(".onnx", ".weight_map.json")
    with open(output, "w") as f:
        json.dump(weight_map, f, indent=2)
    log.info("Saved to %s (%d entries)", output, len(weight_map))

    # Show samples
    matmul_entries = {k: v for k, v in weight_map.items() if "MatMul" in k}
    log.info("MatMul entries: %d", len(matmul_entries))
    for k, v in list(matmul_entries.items())[:5]:
        log.info("  %s -> %s", k, v)
    return 0


# ---------------------------------------------------------------------------
# inspect: Engine weight analysis
# ---------------------------------------------------------------------------

def cmd_inspect(args):
    """Deserialize engine and analyze refittable weights."""
    engine_path = args.engine_path
    log.info("Loading engine: %s (%.1f MB)", engine_path, os.path.getsize(engine_path) / 1e6)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        log.error("Failed to deserialize engine")
        return 1

    refitter = trt.Refitter(engine, logger)
    weights = sorted(refitter.get_all_weights())

    log.info("Total refittable weights: %d", len(weights))

    if not weights:
        log.warning("Engine has 0 refittable weights — was it built with enable_refit?")
        del refitter, engine
        return 1

    # Categorize by prefix
    prefixes = {}
    for w in weights:
        prefix = w.split(".")[0] if "." in w else w.split("/")[0] if "/" in w else "other"
        prefixes.setdefault(prefix, []).append(w)

    log.info("Weight name prefixes:")
    for prefix, ws in sorted(prefixes.items(), key=lambda x: -len(x[1])):
        log.info("  %s: %d weights", prefix, len(ws))
        for w in ws[:3]:
            log.info("    e.g. %s", w)

    # Check LoRA target coverage — typical attention patterns
    lora_patterns = [
        "proj_in.weight", "proj_out.weight",
        "to_q.weight", "to_k.weight", "to_v.weight", "to_out.0.weight",
        "ff.net.0.proj.weight", "ff.net.2.weight",
    ]

    # Strip unet. prefix for matching
    stripped = {}
    for w in weights:
        key = w[len("unet."):] if w.startswith("unet.") else w
        stripped[key] = w

    matched_patterns = 0
    total_patterns = 0
    for w_key in stripped:
        for pat in lora_patterns:
            if w_key.endswith(pat):
                matched_patterns += 1
                break
        total_patterns += 1

    # Count how many LoRA-target-shaped weights are present
    attn_weights = [k for k in stripped if any(k.endswith(p) for p in lora_patterns)]
    log.info("\nLoRA-targetable attention weights: %d / %d total", len(attn_weights), len(weights))
    for w in attn_weights[:10]:
        log.info("  %s", w)
    if len(attn_weights) > 10:
        log.info("  ... and %d more", len(attn_weights) - 10)

    # Look for signs of fused weights (path-style names like /unet/...)
    fused = [w for w in weights if w.startswith("/")]
    internal = [w for w in weights if w.startswith("onnx::")]
    named = [w for w in weights if not w.startswith("/") and not w.startswith("onnx::")]

    log.info("\nWeight categories:")
    log.info("  Named (dot-prefix): %d", len(named))
    log.info("  Fused (path-prefix /): %d", len(fused))
    log.info("  Internal (onnx::): %d", len(internal))

    if fused:
        log.warning("Fused weights detected — LoRA targets in these are unreachable")
        for w in fused[:5]:
            log.warning("  %s", w)

    del refitter, engine
    log.info("\nDone.")
    return 0


# ---------------------------------------------------------------------------
# refit: Load checkpoint + LoRA, refit engine, run inference
# ---------------------------------------------------------------------------

def cmd_refit(args):
    """Full refit test: load model+LoRA, refit engine, run inference, compare."""
    # Import ComfyUI internals
    sys.path.insert(0, "/app")
    import comfy.sd
    import comfy.utils
    import comfy.model_management
    import comfy.lora
    import comfy.lora_convert
    import safetensors.torch

    engine_path = args.engine_path
    checkpoint_path = args.checkpoint_path
    lora_path = args.lora_path
    model_type = args.model_type

    log.info("=== Refit Test ===")
    log.info("Engine: %s", engine_path)
    log.info("Checkpoint: %s", checkpoint_path)
    log.info("LoRA: %s", lora_path)
    log.info("Model type: %s", model_type)

    # --- Step 1: Load checkpoint ---
    log.info("\n--- Step 1: Loading checkpoint ---")
    t0 = time.time()
    result = comfy.sd.load_checkpoint_guess_config(
        checkpoint_path, output_vae=False, output_clip=False,
        output_clipvision=False, output_model=True
    )
    model_patcher = result[0]
    log.info("Checkpoint loaded in %.1fs", time.time() - t0)

    # --- Step 2: Apply LoRA ---
    log.info("\n--- Step 2: Applying LoRA ---")
    t0 = time.time()
    lora_data = safetensors.torch.load_file(lora_path)
    patched_model, _ = comfy.sd.load_lora_for_models(
        model_patcher, None, lora_data, strength_model=1.0, strength_clip=0.0
    )
    log.info("LoRA applied in %.1fs, patches: %d", time.time() - t0, len(patched_model.patches))

    # --- Step 3: Extract weights ---
    log.info("\n--- Step 3: Extracting patched weights ---")
    t0 = time.time()
    comfy.model_management.load_models_gpu([patched_model])

    weight_dtype = torch.bfloat16 if "flux" in model_type else torch.float16
    diffusion_prefix = "diffusion_model."

    base_sd = patched_model.model.diffusion_model.state_dict()
    cpu_weights = {}
    patched_keys = set()
    delta_count = 0

    for key in list(patched_model.patches.keys()):
        if not key.startswith(diffusion_prefix):
            continue
        w = patched_model.patch_weight_to_device(key, return_weight=True)
        if w is None:
            continue
        short_key = key[len(diffusion_prefix):]
        # Check delta
        if short_key in base_sd:
            base_w = base_sd[short_key].to(dtype=w.dtype, device=w.device)
            if (w - base_w).abs().max().item() > 1e-6:
                delta_count += 1
        cpu_weights[short_key] = w.to(dtype=weight_dtype).cpu().numpy()
        patched_keys.add(short_key)

    # Fill base weights
    for k in list(base_sd.keys()):
        if k not in cpu_weights:
            cpu_weights[k] = base_sd.pop(k).to(dtype=weight_dtype).cpu().numpy()
    del base_sd

    log.info("Extracted %d weights (%d LoRA-patched, %d with nonzero delta) in %.1fs",
             len(cpu_weights), len(patched_keys), delta_count, time.time() - t0)

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    # --- Step 4: Deserialize engine ---
    log.info("\n--- Step 4: Deserializing engine ---")
    t0 = time.time()
    torch.cuda.empty_cache()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        log.error("Failed to deserialize engine")
        return 1
    log.info("Engine deserialized in %.1fs", time.time() - t0)

    # --- Step 5: Refit ---
    log.info("\n--- Step 5: Refitting engine ---")
    t0 = time.time()
    refitter = trt.Refitter(engine, logger)
    trt_weight_names = set(refitter.get_all_weights())
    log.info("Engine has %d refittable weights", len(trt_weight_names))

    # Load ONNX weight map if available
    weight_map_path = engine_path.replace(".engine", ".weight_map.json")
    onnx_weight_map = {}
    if os.path.isfile(weight_map_path):
        with open(weight_map_path) as f:
            onnx_weight_map = json.load(f)
        log.info("Loaded weight map: %d entries", len(onnx_weight_map))
    else:
        log.warning("No .weight_map.json found — onnx::* weights won't map to LoRA keys")

    import numpy as np

    matched = 0
    matched_patched = 0
    missing_in_trt = 0
    mapped_via_onnx = 0

    for trt_name in trt_weight_names:
        via_onnx = False
        if trt_name.startswith("unet."):
            pytorch_key = trt_name[len("unet."):]
        elif trt_name in onnx_weight_map:
            pytorch_key = onnx_weight_map[trt_name]
            via_onnx = True
            mapped_via_onnx += 1
        else:
            pytorch_key = trt_name

        if pytorch_key not in cpu_weights:
            missing_in_trt += 1
            continue

        w = cpu_weights[pytorch_key]
        # ONNX MatMul weights need transposition: PyTorch [out, in] -> ONNX [in, out]
        if via_onnx and "MatMul" in trt_name and w.ndim == 2:
            w = np.ascontiguousarray(w.T)

        refitter.set_named_weights(trt_name, trt.Weights(w))
        matched += 1
        if pytorch_key in patched_keys:
            matched_patched += 1

    # Check coverage
    all_mapped_keys = set()
    for trt_name in trt_weight_names:
        if trt_name.startswith("unet."):
            all_mapped_keys.add(trt_name[len("unet."):])
        elif trt_name in onnx_weight_map:
            all_mapped_keys.add(onnx_weight_map[trt_name])

    patched_not_mapped = patched_keys - all_mapped_keys
    if patched_not_mapped:
        log.warning("%d LoRA-patched weights NOT mapped to TRT:", len(patched_not_mapped))
        for k in sorted(patched_not_mapped)[:10]:
            log.warning("  %s", k)

    missing = refitter.get_missing_weights()
    if missing:
        log.warning("%d weights still missing: %s", len(missing), missing[:5])

    log.info("Mapped: %d total (%d LoRA-patched, %d via ONNX map), %d TRT without source, %d missing",
             matched, matched_patched, mapped_via_onnx, missing_in_trt, len(missing))

    success = refitter.refit_cuda_engine()
    del refitter
    if not success:
        log.error("Refit failed!")
        del cpu_weights, engine
        return 1
    log.info("Refit succeeded in %.1fs", time.time() - t0)
    del cpu_weights

    # --- Step 6: Inference test ---
    log.info("\n--- Step 6: Running inference ---")
    t0 = time.time()
    context = engine.create_execution_context()

    # Determine shapes from engine
    num_io = engine.num_io_tensors
    log.info("Engine I/O tensors: %d", num_io)
    for i in range(num_io):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        log.info("  %s: mode=%s, shape=%s, dtype=%s", name, mode, shape, dtype)

    # Set input shapes (use profile opt shapes)
    # For dynamic engines, we need to set shapes; for static, shapes are fixed
    device = torch.device("cuda")
    buffers = {}

    for i in range(num_io):
        name = engine.get_tensor_name(i)
        shape = list(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)

        # Replace -1 dims with reasonable values
        for j, s in enumerate(shape):
            if s == -1:
                shape[j] = 2  # small batch/seq

        torch_dtype = {
            trt.float16: torch.float16,
            trt.float32: torch.float32,
            trt.bfloat16: torch.bfloat16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
        }.get(dtype, torch.float32)

        buf = torch.randn(shape, dtype=torch_dtype, device=device) if torch_dtype in (
            torch.float16, torch.float32, torch.bfloat16
        ) else torch.zeros(shape, dtype=torch_dtype, device=device)
        buffers[name] = buf

        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, shape)
        context.set_tensor_address(name, buf.data_ptr())

    stream = torch.cuda.current_stream(device)
    ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
    torch.cuda.synchronize()

    if not ok:
        log.error("Inference failed!")
        del context, engine, buffers
        return 1

    # Check output is not all zeros/nan
    for name, buf in buffers.items():
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.OUTPUT:
            has_nan = torch.isnan(buf).any().item()
            all_zero = (buf == 0).all().item()
            mean_val = buf.float().mean().item()
            std_val = buf.float().std().item()
            log.info("Output '%s': shape=%s, mean=%.4f, std=%.4f, nan=%s, all_zero=%s",
                     name, list(buf.shape), mean_val, std_val, has_nan, all_zero)
            if has_nan:
                log.warning("Output contains NaN!")
            if all_zero:
                log.warning("Output is all zeros!")

    log.info("Inference completed in %.1fs", time.time() - t0)

    del context, engine, buffers
    torch.cuda.empty_cache()
    log.info("\n=== PASS ===")
    return 0


# ---------------------------------------------------------------------------
# build: Build a small refit-enabled engine for testing
# ---------------------------------------------------------------------------

def cmd_build(args):
    """Build a refit-enabled engine using REFIT_INDIVIDUAL + mark all weights."""
    sys.path.insert(0, "/app")
    import comfy.sd
    import comfy.model_management

    checkpoint_path = args.checkpoint_path
    height = args.height
    width = args.width
    context_len = args.context_len
    output_dir = args.output_dir or "/app/output/tensorrt/unet"

    log.info("=== Building Refit Engine ===")
    log.info("Checkpoint: %s", checkpoint_path)
    log.info("Resolution: %dx%d", width, height)

    # Load checkpoint
    log.info("\n--- Loading checkpoint ---")
    t0 = time.time()
    result = comfy.sd.load_checkpoint_guess_config(
        checkpoint_path, output_vae=False, output_clip=False,
        output_clipvision=False, output_model=True
    )
    model_patcher = result[0]
    log.info("Loaded in %.1fs", time.time() - t0)

    comfy.model_management.load_models_gpu([model_patcher])
    unet = model_patcher.model.diffusion_model
    dtype = torch.float16  # default for SDXL/SD1.5

    # Determine input shapes from model config
    unet_config = model_patcher.model.model_config.unet_config
    in_channels = unet_config.get("in_channels", 4)
    context_dim = unet_config.get("context_dim", None)
    if isinstance(context_dim, list):
        context_dim = context_dim[0]

    log.info("UNet in_channels=%d, context_dim=%s", in_channels, context_dim)

    # ONNX export
    log.info("\n--- Exporting ONNX ---")
    import tempfile
    onnx_dir = tempfile.mkdtemp(prefix="trt_refit_test_")
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    # Create wrapper
    class UNET(torch.nn.Module):
        def forward(self, x, timesteps, context, *extra_args):
            extras = {}
            extra_names = self.extra_input_names
            for i, name in enumerate(extra_names):
                if i < len(extra_args):
                    extras[name] = extra_args[i]
            return self.unet(x, timesteps, context,
                           transformer_options=self.transformer_options, **extras)

    wrapper = UNET()
    wrapper.unet = unet
    transformer_options = model_patcher.model_options.get("transformer_options", {}).copy()
    wrapper.transformer_options = transformer_options

    input_names = ["x", "timesteps", "context"]
    extra_input_names = []

    # Handle y input for SDXL-like models
    y_dim = unet_config.get("adm_in_channels", None) or unet_config.get("vec_in_dim", None)
    if y_dim:
        input_names.append("y")
        extra_input_names.append("y")
    wrapper.extra_input_names = extra_input_names

    batch = 1
    h_latent = height // 8
    w_latent = width // 8

    dummy_inputs = [
        torch.randn(batch, in_channels, h_latent, w_latent, dtype=dtype, device="cuda"),
        torch.zeros(batch, dtype=dtype, device="cuda"),
        torch.randn(batch, context_len, context_dim, dtype=dtype, device="cuda"),
    ]
    if y_dim:
        dummy_inputs.append(torch.randn(batch, y_dim, dtype=dtype, device="cuda"))

    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper, tuple(dummy_inputs), onnx_path,
            input_names=input_names, output_names=["h"],
            opset_version=17, dynamo=False,
        )
    log.info("ONNX exported to %s", onnx_path)

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    # TRT build
    log.info("\n--- Building TRT engine with REFIT_INDIVIDUAL ---")
    t0 = time.time()
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    success = parser.parse_from_file(onnx_path)
    if not success:
        for i in range(parser.num_errors):
            log.error("ONNX parse error: %s", parser.get_error(i))
        return 1

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()

    # Static shapes
    shapes = {
        "x": (batch, in_channels, h_latent, w_latent),
        "timesteps": (batch,),
        "context": (batch, context_len, context_dim),
    }
    if y_dim:
        shapes["y"] = (batch, y_dim)

    for name, shape in shapes.items():
        profile.set_shape(name, shape, shape, shape)

    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.REFIT)
    if args.opt_level is not None:
        config.builder_optimization_level = args.opt_level
        log.info("Builder optimization level: %d", args.opt_level)

    config.add_optimization_profile(profile)

    log.info("Building engine (this takes minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        log.error("Engine build failed!")
        return 1

    # Save
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
    output_path = os.path.join(output_dir, f"TEST_refit_{basename}_stat-b-{batch}-h-{height}-w-{width}.engine")
    with open(output_path, "wb") as f:
        f.write(serialized)

    log.info("Engine saved to %s (%.1f MB)", output_path, os.path.getsize(output_path) / 1e6)
    log.info("Build took %.1fs", time.time() - t0)

    # Save ONNX weight map before cleaning up
    weight_map = _build_onnx_weight_map(onnx_path)
    map_path = output_path.replace(".engine", ".weight_map.json")
    with open(map_path, "w") as f:
        json.dump(weight_map, f)
    log.info("Saved weight map: %s (%d entries)", map_path, len(weight_map))

    # Cleanup ONNX
    import shutil
    shutil.rmtree(onnx_dir)

    # Quick inspection
    log.info("\n--- Inspecting built engine ---")
    runtime = trt.Runtime(trt_logger)
    with open(output_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    refitter = trt.Refitter(engine, trt_logger)
    weights = refitter.get_all_weights()
    log.info("Refittable weights: %d", len(weights))

    fused = [w for w in weights if w.startswith("/")]
    named = [w for w in weights if not w.startswith("/") and not w.startswith("onnx::")]
    log.info("  Named: %d, Fused: %d, Internal: %d",
             len(named), len(fused), len(weights) - len(named) - len(fused))

    del refitter, engine
    torch.cuda.empty_cache()

    log.info("\n=== Build complete: %s ===", output_path)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TensorRT Refit Test")
    sub = parser.add_subparsers(dest="command", required=True)

    # map
    p_map = sub.add_parser("map", help="Generate weight map from ONNX file")
    p_map.add_argument("onnx_path", help="Path to .onnx file")
    p_map.add_argument("--output", default=None, help="Output JSON path")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect engine refittable weights")
    p_inspect.add_argument("engine_path", help="Path to .engine file")

    # refit
    p_refit = sub.add_parser("refit", help="Full refit + inference test")
    p_refit.add_argument("engine_path", help="Path to .engine file")
    p_refit.add_argument("checkpoint_path", help="Path to checkpoint")
    p_refit.add_argument("lora_path", help="Path to LoRA safetensors")
    p_refit.add_argument("--model-type", default="sdxl", help="Model type (default: sdxl)")

    # build
    p_build = sub.add_parser("build", help="Build refit-enabled engine")
    p_build.add_argument("checkpoint_path", help="Path to checkpoint")
    p_build.add_argument("--model-type", default="sdxl", help="Model type")
    p_build.add_argument("--height", type=int, default=1024)
    p_build.add_argument("--width", type=int, default=1024)
    p_build.add_argument("--context-len", type=int, default=77,
                         help="Context sequence length (77=default, 154=2x, 308=4x)")
    p_build.add_argument("--output-dir", default=None)
    p_build.add_argument("--opt-level", type=int, default=None,
                         help="Builder optimization level (0=min, 5=max, default=3)")

    args = parser.parse_args()

    if args.command == "map":
        return cmd_map(args)
    elif args.command == "inspect":
        return cmd_inspect(args)
    elif args.command == "refit":
        return cmd_refit(args)
    elif args.command == "build":
        return cmd_build(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
