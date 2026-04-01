"""TensorRT Loader Auto — combined build + load + refit node."""

import json
import logging
import os
import re

import folder_paths
from server import PromptServer

log = logging.getLogger("comfyui_tensorrt")


def _send_trt_progress(phase):
    """Send progress event to frontend toast."""
    PromptServer.instance.send_sync("trt_auto_progress", {"phase": phase})


# Model types supported by the auto node (no SVD/Flux yet)
AUTO_MODEL_TYPES = [
    "sdxl_base",
    "sdxl_inpaint",
    "sdxl_refiner",
    "sd1.x",
    "sd2.x-768v",
]

# Engine cache: skip redundant refit when patches haven't changed
_refit_cache = {
    "patches_uuid": None,
    "engine_path": None,
    "patcher": None,
}

# Engine load cache: skip re-probing when engine path hasn't changed
_engine_cache = {
    "engine_path": None,
    "patcher": None,
}


def _auto_engine_dir():
    """Return the auto-managed engine directory path."""
    return os.path.join(folder_paths.models_dir, "tensorrt", "auto")


def _make_engine_filename(prefix, profile_desc):
    """Build the deterministic engine filename (no counter)."""
    return f"{prefix}_refit_${profile_desc}.engine"


def _find_by_profile(directory, profile_desc):
    """Find the newest engine matching a profile desc in a directory."""
    if not os.path.isdir(directory):
        return None
    pattern = re.compile(re.escape(f"_refit_${profile_desc}"))
    candidates = []
    for f in os.listdir(directory):
        if f.endswith(".engine") and pattern.search(f):
            full = os.path.join(directory, f)
            candidates.append((os.path.getmtime(full), full))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _symlink_into_auto(source, auto_dir):
    """Symlink an engine (and its weight map sidecar) into the auto dir."""
    dest = os.path.join(auto_dir, os.path.basename(source))
    if os.path.exists(dest):
        return dest
    try:
        os.symlink(os.path.abspath(source), dest)
    except OSError as e:
        log.warning("Auto: symlink failed (%s), copying instead", e)
        import shutil

        shutil.copy2(source, dest)

    source_map = source.replace(".engine", ".weight_map.json")
    if os.path.isfile(source_map):
        dest_map = dest.replace(".engine", ".weight_map.json")
        if not os.path.exists(dest_map):
            try:
                os.symlink(os.path.abspath(source_map), dest_map)
            except OSError:
                import shutil

                shutil.copy2(source_map, dest_map)

    return dest


def _find_existing_engine(profile_desc):
    """Search for a matching engine by profile, checking auto/ first, then output/tensorrt/unet/.

    Detection is profile-based — filename prefix is irrelevant.
    If found in output dir, symlinks it into auto/. Returns the engine path or None.
    """
    auto_dir = _auto_engine_dir()
    os.makedirs(auto_dir, exist_ok=True)

    # Check auto/ dir first
    found = _find_by_profile(auto_dir, profile_desc)
    if found:
        log.info("Auto: found existing engine: %s", found)
        return found

    # Search output/tensorrt/unet/
    output_unet_dir = os.path.join(
        folder_paths.get_output_directory(), "tensorrt", "unet"
    )
    found = _find_by_profile(output_unet_dir, profile_desc)
    if found:
        log.info("Auto: found matching engine in output dir, symlinking: %s", found)
        return _symlink_into_auto(found, auto_dir)

    log.info("Auto: no existing engine found for profile %s", profile_desc)
    return None


def _fifo_evict(auto_dir, max_bytes, estimated_new_bytes=0):
    """Evict oldest real (non-symlink) engines until under budget."""
    if not os.path.isdir(auto_dir):
        return

    entries = []
    total_real = 0
    for f in os.listdir(auto_dir):
        if not f.endswith(".engine"):
            continue
        path = os.path.join(auto_dir, f)
        if os.path.islink(path):
            continue
        size = os.path.getsize(path)
        entries.append((os.path.getmtime(path), path, size))
        total_real += size

    if total_real + estimated_new_bytes <= max_bytes:
        return

    # Sort oldest first
    entries.sort()
    for _mtime, path, size in entries:
        if total_real + estimated_new_bytes <= max_bytes:
            break
        log.info("Auto: FIFO evicting %s (%.1f MB)", path, size / (1024 * 1024))
        os.remove(path)
        # Also remove sidecar
        sidecar = path.replace(".engine", ".weight_map.json")
        if os.path.isfile(sidecar) and not os.path.islink(sidecar):
            os.remove(sidecar)
        total_real -= size


def _do_refit(engine, unet_path, source_model, model_type):
    """Refit engine weights from source_model. Returns True on success."""
    import numpy as np

    import comfy.model_management
    import tensorrt as trt

    trt_log = trt.Logger(trt.Logger.INFO)

    weight_dtype = __import__("torch").float16
    if model_type in ("flux_dev", "flux_schnell"):
        weight_dtype = __import__("torch").bfloat16

    # Extract LoRA-patched weights
    log.info("Refit: loading source model to extract weights...")
    comfy.model_management.load_models_gpu([source_model])

    base_sd = source_model.model.diffusion_model.state_dict()
    diffusion_prefix = "diffusion_model."
    cpu_weights = {}
    patched_keys = set()
    delta_nonzero = 0

    for key in list(source_model.patches.keys()):
        if not key.startswith(diffusion_prefix):
            continue
        w = source_model.patch_weight_to_device(key, return_weight=True)
        if w is None:
            continue
        short_key = key[len(diffusion_prefix) :]
        if short_key in base_sd:
            base_w = base_sd[short_key].to(dtype=w.dtype, device=w.device)
            if (w - base_w).abs().max().item() > 1e-6:
                delta_nonzero += 1
        cpu_weights[short_key] = w.to(dtype=weight_dtype).cpu().numpy()
        patched_keys.add(short_key)

    # Fill base weights
    for k in list(base_sd.keys()):
        if k not in cpu_weights:
            cpu_weights[k] = base_sd.pop(k).to(dtype=weight_dtype).cpu().numpy()
    del base_sd

    log.info(
        "Refit: %d patched weights, %d differ from base",
        len(patched_keys),
        delta_nonzero,
    )

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    # Load weight map sidecar
    weight_map_path = unet_path.replace(".engine", ".weight_map.json")
    onnx_weight_map = {}
    if os.path.isfile(weight_map_path):
        with open(weight_map_path) as f:
            onnx_weight_map = json.load(f)
        log.info("Refit: loaded weight map with %d entries", len(onnx_weight_map))
    else:
        log.warning("Refit: no .weight_map.json sidecar — onnx::* weights unmapped")

    # Apply weights
    refitter = trt.Refitter(engine, trt_log)
    trt_weight_names = set(refitter.get_all_weights())
    log.info("Refit: TRT has %d refittable weights", len(trt_weight_names))

    matched = 0
    for trt_name in trt_weight_names:
        via_onnx = False
        if trt_name.startswith("unet."):
            pytorch_key = trt_name[len("unet.") :]
        elif trt_name in onnx_weight_map:
            pytorch_key = onnx_weight_map[trt_name]
            via_onnx = True
        else:
            continue

        if pytorch_key not in cpu_weights:
            continue

        w = cpu_weights[pytorch_key]
        if via_onnx and "MatMul" in trt_name and w.ndim == 2:
            w = np.ascontiguousarray(w.T)

        refitter.set_named_weights(trt_name, trt.Weights(w))
        matched += 1

    log.info("Refit: %d weights mapped", matched)

    missing = refitter.get_missing_weights()
    if missing:
        log.warning("Refit: %d weights still missing: %s", len(missing), missing[:5])

    success = refitter.refit_cuda_engine()
    del refitter, cpu_weights
    if not success:
        raise RuntimeError("TensorRT engine refit failed.")
    log.info("Refit: engine weights updated successfully")
    return True


class TensorRTLoaderAuto:
    """Auto-managed TRT engine: build if absent, load, and optionally refit LoRA weights."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Source model from a checkpoint loader, optionally with LoRAs applied. "
                        "With refit=True, only base weights are used for building — LoRAs are applied at load time. "
                        "With refit=False, all applied weights (including LoRAs) are baked into the engine permanently.",
                    },
                ),
                "model_type": (
                    AUTO_MODEL_TYPES,
                    {
                        "tooltip": "Architecture of the model. Must match the checkpoint.",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "AUTO_{modelname}",
                        "tooltip": "Engine filename prefix. {modelname} is auto-resolved from the upstream checkpoint loader.",
                    },
                ),
                "refit": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, the engine is built from base weights only — LoRA/patch deltas "
                        "are applied into the TRT engine at load time (~13s for SDXL). "
                        "When disabled, all applied weights (including LoRAs) are baked into the engine permanently.",
                    },
                ),
                "on_missing": (
                    ["build", "error"],
                    {
                        "default": "build",
                        "tooltip": "What to do when no matching engine exists. "
                        "'build': automatically build the engine (5-10 min for SDXL), shows build configuration widgets. "
                        "'error': raise an error — use this when engines are pre-built externally.",
                    },
                ),
                "static_shapes": (
                    ["static", "dynamic"],
                    {
                        "default": "static",
                        "tooltip": "Static: fixed dimensions, best performance. Dynamic: flexible resolution range, slightly slower.",
                    },
                ),
                "context_len": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": "CLIP context multiplier. 4 = 308 tokens (SDXL with prompt weighting). "
                        "1 = 77 tokens (standard). Must cover your longest prompt or inference will fail.",
                    },
                ),
                # Static shape widgets
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Fixed height in pixels (static mode).",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Fixed width in pixels (static mode).",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Fixed batch size (static mode).",
                    },
                ),
                # Dynamic shape widgets
                "min_height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Minimum height (dynamic mode).",
                    },
                ),
                "opt_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Optimal height — best TRT performance at this value.",
                    },
                ),
                "max_height": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Maximum height (dynamic mode). Wider range = more VRAM.",
                    },
                ),
                "min_width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Minimum width (dynamic mode).",
                    },
                ),
                "opt_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Optimal width — best TRT performance at this value.",
                    },
                ),
                "max_width": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Maximum width (dynamic mode). Wider range = more VRAM.",
                    },
                ),
                "min_batch": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Minimum batch size (dynamic mode).",
                    },
                ),
                "opt_batch": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Optimal batch size (dynamic mode).",
                    },
                ),
                "max_batch": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Maximum batch size (dynamic mode).",
                    },
                ),
                # Disk management
                "disk_management": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable automatic disk usage management for models/tensorrt/auto/. "
                        "Only engines built by this node are affected — manually built engines in other directories are never touched. "
                        "When the auto directory exceeds the size limit, the oldest engines are evicted (FIFO).",
                    },
                ),
                "max_disk_usage_gb": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 1.0,
                        "max": 1000.0,
                        "step": 1.0,
                        "tooltip": "Maximum disk usage in GB for models/tensorrt/auto/. "
                        "Only real files count — symlinked engines from other directories are excluded from both size calculation and eviction.",
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute — internal caches handle skipping redundant work
        return float("NaN")

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "TensorRT"
    DESCRIPTION = (
        "All-in-one TensorRT UNet: auto-builds engine if absent, loads it, "
        "and optionally refits LoRA weights from the source model. "
        "Engines are stored in models/tensorrt/auto/ with optional FIFO disk management."
    )

    def execute(
        self,
        model,
        model_type,
        filename_prefix,
        refit,
        on_missing,
        static_shapes,
        context_len,
        height,
        width,
        batch_size,
        min_height,
        opt_height,
        max_height,
        min_width,
        opt_width,
        max_width,
        min_batch,
        opt_batch,
        max_batch,
        disk_management,
        max_disk_usage_gb,
        prompt=None,
        unique_id=None,
    ):
        global _refit_cache, _engine_cache
        import comfy.utils
        from .tensorrt_convert import (
            _derive_model_name,
            _make_profile_desc,
            build_unet_engine,
        )
        from .tensorrt_loader import TrTUnet, _create_model_for_type, _wrap_trt_patcher

        pbar = comfy.utils.ProgressBar(4)

        # Resolve {modelname} placeholder
        if "{modelname}" in filename_prefix:
            modelname = _derive_model_name(prompt, unique_id, "model") or "model"
            filename_prefix = filename_prefix.replace("{modelname}", modelname)

        # Build profile description and expected engine path
        if static_shapes == "static":
            profile_desc = _make_profile_desc(
                True,
                batch_size,
                batch_size,
                batch_size,
                height,
                height,
                height,
                width,
                width,
                width,
                context_len=context_len,
            )
        else:
            profile_desc = _make_profile_desc(
                False,
                min_batch,
                opt_batch,
                max_batch,
                min_height,
                opt_height,
                max_height,
                min_width,
                opt_width,
                max_width,
                context_len=context_len,
            )

        auto_dir = _auto_engine_dir()
        os.makedirs(auto_dir, exist_ok=True)

        # Step 1: Find existing engine by profile (prefix-independent)
        pbar.update_absolute(0, 4)
        _send_trt_progress("searching")
        engine_path = _find_existing_engine(profile_desc)

        if engine_path is None:
            if on_missing == "error":
                raise FileNotFoundError(
                    f"No TRT engine found matching profile {profile_desc} and on_missing='error'. "
                    "Set on_missing to 'build' or build the engine manually first."
                )

            # Use prefix only for naming newly built engines
            engine_filename = _make_engine_filename(filename_prefix, profile_desc)
            engine_path = os.path.join(auto_dir, engine_filename)

            log.info("Auto: building engine — this takes 5-10 minutes for SDXL...")
            _send_trt_progress("building")

            if disk_management:
                # Estimate ~2GB for SDXL static engine
                _fifo_evict(
                    auto_dir,
                    int(max_disk_usage_gb * 1024**3),
                    estimated_new_bytes=2 * 1024**3,
                )

            if static_shapes == "static":
                build_unet_engine(
                    model,
                    engine_path,
                    batch_size,
                    batch_size,
                    batch_size,
                    height,
                    height,
                    height,
                    width,
                    width,
                    width,
                    context_len,
                    context_len,
                    context_len,
                    num_video_frames=0,
                    is_static=True,
                    enable_refit=True,
                )
            else:
                build_unet_engine(
                    model,
                    engine_path,
                    min_batch,
                    opt_batch,
                    max_batch,
                    min_height,
                    opt_height,
                    max_height,
                    min_width,
                    opt_width,
                    max_width,
                    1,
                    context_len,
                    context_len,
                    num_video_frames=0,
                    is_static=False,
                    enable_refit=True,
                )

        # Step 2: Check caches
        pbar.update_absolute(1, 4)
        has_patches = refit and hasattr(model, "patches") and len(model.patches) > 0
        current_uuid = getattr(model, "patches_uuid", None) if has_patches else None

        if (
            has_patches
            and current_uuid is not None
            and _refit_cache["patches_uuid"] == current_uuid
            and _refit_cache["engine_path"] == engine_path
            and _refit_cache["patcher"] is not None
        ):
            log.info("Auto: refit cache hit (patches_uuid unchanged), skipping refit")
            _send_trt_progress("cached")
            return (_refit_cache["patcher"],)

        if (
            not has_patches
            and _engine_cache["engine_path"] == engine_path
            and _engine_cache["patcher"] is not None
        ):
            log.info("Auto: engine cache hit (no refit needed)")
            _send_trt_progress("cached")
            return (_engine_cache["patcher"],)

        # Step 3: Load engine
        pbar.update_absolute(2, 4)
        _send_trt_progress("loading")
        import torch

        torch.cuda.empty_cache()
        unet = TrTUnet(engine_path)
        model_shell = _create_model_for_type(model_type, unet)
        patcher = _wrap_trt_patcher(model_shell, unet)

        # Step 4: Refit if needed
        pbar.update_absolute(3, 4)
        if has_patches:
            log.info("Auto: refitting with LoRA patches...")
            _send_trt_progress("refitting")
            import tensorrt as trt

            trt_runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            with open(engine_path, "rb") as f:
                engine = trt_runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

            _do_refit(engine, engine_path, model, model_type)

            # Create new patcher from refitted engine
            from .tensorrt_loader import TrTUnet as _TrTUnet

            unet = _TrTUnet.from_engine(engine)
            model_shell = _create_model_for_type(model_type, unet)
            patcher = _wrap_trt_patcher(model_shell, unet)

            _refit_cache["patches_uuid"] = current_uuid
            _refit_cache["engine_path"] = engine_path
            _refit_cache["patcher"] = patcher
        else:
            _engine_cache["engine_path"] = engine_path
            _engine_cache["patcher"] = patcher

        pbar.update_absolute(4, 4)
        _send_trt_progress("done")
        return (patcher,)
