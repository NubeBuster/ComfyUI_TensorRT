"""TensorRT Loader Auto — combined build + load + refit node."""

import datetime
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


def _refit_cache_dir():
    """Return the hidden refit cache directory path."""
    d = os.path.join(_auto_engine_dir(), ".refit_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _refit_cache_path(engine_path):
    """Return the path for a persisted refitted engine."""
    stem = os.path.basename(engine_path)
    return os.path.join(_refit_cache_dir(), stem)


def _make_engine_filename(prefix, profile_desc):
    """Build the deterministic engine filename (no counter)."""
    return f"{prefix}_refit_${profile_desc}.engine"


def _find_by_profile(directory, profile_desc, model_name=None):
    """Find the newest engine matching a profile desc in a directory.

    If model_name is provided, only matches engines whose .meta.json sidecar
    has a matching model_name. This prevents cross-model collisions when
    multiple checkpoints share the same resolution/batch/context profile.
    """
    if not os.path.isdir(directory):
        return None
    pattern = re.compile(re.escape(f"_refit_${profile_desc}"))
    candidates = []
    for f in os.listdir(directory):
        if not f.endswith(".engine") or not pattern.search(f):
            continue
        full = os.path.join(directory, f)
        if model_name:
            meta = _read_meta_sidecar(full)
            if meta.get("model_name") != model_name:
                continue
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


def _write_meta_sidecar(engine_path, model_type, model_name, profile_desc, refit):
    """Write a .meta.json sidecar alongside an engine file."""
    meta_path = engine_path.replace(".engine", ".meta.json")
    meta = {
        "model_type": model_type,
        "model_name": model_name,
        "profile_desc": profile_desc,
        "refit": refit,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    except OSError as e:
        log.warning("Auto: failed to write meta sidecar: %s", e)


def _read_meta_sidecar(engine_path):
    """Read .meta.json sidecar if it exists, else return empty dict."""
    meta_path = engine_path.replace(".engine", ".meta.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _collect_auto_models_info(auto_dir):
    """Gather info on all engines in the auto directory."""
    models = []
    if not os.path.isdir(auto_dir):
        return models
    for f in sorted(os.listdir(auto_dir)):
        if not f.endswith(".engine"):
            continue
        path = os.path.join(auto_dir, f)
        is_link = os.path.islink(path)
        try:
            stat = os.stat(path)
            disk_size = stat.st_size
            created_at = datetime.datetime.fromtimestamp(
                stat.st_mtime, tz=datetime.timezone.utc
            ).isoformat()
        except OSError:
            disk_size = 0
            created_at = None
        meta = _read_meta_sidecar(path)
        models.append(
            {
                "filename": f,
                "path": path,
                "is_symlink": is_link,
                "disk_size_bytes": disk_size if not is_link else 0,
                "created_at": created_at,
                **meta,
            }
        )
    return models


def _find_existing_engine(profile_desc, model_name=None):
    """Search for a matching engine by profile + model name (from .meta.json sidecar).

    Checks auto/ first, then output/tensorrt/unet/.
    If found in output dir, symlinks it into auto/. Returns the engine path or None.
    """
    auto_dir = _auto_engine_dir()
    os.makedirs(auto_dir, exist_ok=True)

    # Check auto/ dir first
    found = _find_by_profile(auto_dir, profile_desc, model_name)
    if found:
        log.info("Auto: found existing engine: %s", found)
        return found

    # Search output/tensorrt/unet/
    output_unet_dir = os.path.join(
        folder_paths.get_output_directory(), "tensorrt", "unet"
    )
    found = _find_by_profile(output_unet_dir, profile_desc, model_name)
    if found:
        log.info("Auto: found matching engine in output dir, symlinking: %s", found)
        return _symlink_into_auto(found, auto_dir)

    log.info(
        "Auto: no existing engine found for model=%s profile=%s",
        model_name,
        profile_desc,
    )
    return None


def _list_real_engines(auto_dir):
    """List real (non-symlink) engine files sorted oldest first."""
    entries = []
    total_real = 0
    if not os.path.isdir(auto_dir):
        return entries, total_real
    for f in os.listdir(auto_dir):
        if not f.endswith(".engine"):
            continue
        path = os.path.join(auto_dir, f)
        if os.path.islink(path):
            continue
        size = os.path.getsize(path)
        entries.append((os.path.getmtime(path), path, size))
        total_real += size
    entries.sort()
    return entries, total_real


def _evict_engine(path, size):
    """Remove an engine file, its sidecars, and refit cache. Returns (filename, size_bytes)."""
    log.info("Auto: FIFO evicting %s (%.1f MB)", path, size / (1024 * 1024))
    os.remove(path)
    for suffix in (".weight_map.json", ".meta.json"):
        sidecar = path.replace(".engine", suffix)
        if os.path.isfile(sidecar) and not os.path.islink(sidecar):
            os.remove(sidecar)
    # Clean up persisted refit cache
    cached = _refit_cache_path(path)
    if os.path.isfile(cached):
        os.remove(cached)
    return (os.path.basename(path), size)


def _purge_refit_cache():
    """Remove all persisted refitted engines. Returns total bytes freed."""
    cache_dir = _refit_cache_dir()
    freed = 0
    for f in os.listdir(cache_dir):
        if not f.endswith(".engine"):
            continue
        path = os.path.join(cache_dir, f)
        size = os.path.getsize(path)
        log.info("Auto: purging refit cache %s (%.1f MB)", f, size / (1024 * 1024))
        os.remove(path)
        freed += size
    return freed


def _fifo_evict_max_usage(auto_dir, max_bytes, estimated_new_bytes=0):
    """Evict refit cache first, then oldest base engines until under max_bytes. Returns list of (filename, size_bytes)."""
    # Refit cache is expendable — purge it first
    entries, total_real = _list_real_engines(auto_dir)
    if total_real + estimated_new_bytes > max_bytes:
        _purge_refit_cache()
    evicted = []
    for _mtime, path, size in entries:
        if total_real + estimated_new_bytes <= max_bytes:
            break
        evicted.append(_evict_engine(path, size))
        total_real -= size
    return evicted


def _fifo_evict_min_free(auto_dir, min_free_bytes, estimated_new_bytes=0):
    """Evict refit cache first, then oldest base engines until drive has min_free_bytes free. Returns list of (filename, size_bytes)."""
    # Refit cache is expendable — purge it first
    stat = os.statvfs(auto_dir)
    free = stat.f_bavail * stat.f_frsize
    if free - estimated_new_bytes < min_free_bytes:
        _purge_refit_cache()
    entries, _ = _list_real_engines(auto_dir)
    evicted = []
    for _mtime, path, size in entries:
        stat = os.statvfs(auto_dir)
        free = stat.f_bavail * stat.f_frsize
        if free - estimated_new_bytes >= min_free_bytes:
            break
        evicted.append(_evict_engine(path, size))
    return evicted


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
                        "The checkpoint name is auto-detected by walking the graph back through "
                        "LoRA loaders, model merges, etc. to find the source CheckpointLoaderSimple. "
                        "Engines are matched to checkpoints by name — different checkpoints with "
                        "the same resolution get separate engines. "
                        "With refit=True, only base weights are used for building — LoRAs are refitted at load time. "
                        "With refit=False, all applied weights (including LoRAs) are baked in permanently.",
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
                        "tooltip": "Engine filename prefix. {modelname} is auto-resolved from the upstream "
                        "checkpoint loader by walking back through the graph. "
                        "The prefix is only used for naming new engines — existing engines are matched "
                        "by checkpoint name and profile, not by prefix.",
                    },
                ),
                "refit": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, the engine is built from base weights only — LoRA/patch deltas "
                        "are refitted into the TRT engine at load time (~13s for SDXL). "
                        "Refitted engines are cached to disk so subsequent runs skip refitting "
                        "even after VRAM eviction. Cache is invalidated when LoRA patches change. "
                        "When disabled, all applied weights (including LoRAs) are baked into the engine permanently.",
                    },
                ),
                "on_missing": (
                    ["build", "error"],
                    {
                        "default": "build",
                        "tooltip": "What to do when no matching engine exists for this checkpoint + profile. "
                        "'build': automatically build the engine (5-10 min for SDXL). "
                        "'error': raise an error — use this when engines are pre-built externally.",
                    },
                ),
                "static_shapes": (
                    ["static", "dynamic"],
                    {
                        "default": "static",
                        "tooltip": "Static: fixed height/width/batch, best performance. "
                        "Dynamic: flexible resolution range, slightly slower. "
                        "Context length is always dynamic regardless of this setting — "
                        "prompts shorter than context_len will work without rebuilding.",
                    },
                ),
                "context_len": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Maximum CLIP context multiplier. 4 = up to 308 tokens (SDXL with prompt weighting). "
                        "1 = up to 77 tokens (standard). Context is always dynamic — shorter prompts work fine, "
                        "but prompts exceeding this limit will error. Changing this value requires a rebuild.",
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
                    ["disabled", "max_disk_usage", "min_disk_free"],
                    {
                        "default": "disabled",
                        "tooltip": "Disk management for models/tensorrt/auto/. "
                        "Eviction order: refit cache (expendable, ~13s to rebuild) is purged first, "
                        "then oldest base engines (expensive, 5-10 min to rebuild) by FIFO. "
                        "'disabled': no eviction. "
                        "'max_disk_usage': evict when auto/ total size exceeds threshold_gb. "
                        "'min_disk_free': evict when free space on the auto/ drive drops below threshold_gb. "
                        "Only real files are evicted — symlinked engines are excluded.",
                    },
                ),
                "threshold_gb": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 1.0,
                        "max": 1000.0,
                        "step": 1.0,
                        "tooltip": "Threshold in GB. "
                        "For 'max_disk_usage': maximum total size of models/tensorrt/auto/. "
                        "For 'min_disk_free': minimum free space on the drive where auto/ lives. "
                        "Only real files count — symlinked engines are excluded from size calculation and eviction.",
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

    def check_lazy_status(
        self,
        model,
        refit,
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
        **kwargs,
    ):
        """Skip upstream model evaluation when engine exists and refit is off."""
        if refit:
            return ["model"]
        from .tensorrt_convert import _make_profile_desc

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
        if _find_existing_engine(profile_desc) is not None:
            return []
        return ["model"]

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "execute"
    CATEGORY = "TensorRT"
    DESCRIPTION = (
        "All-in-one TensorRT UNet: auto-builds engine if absent, loads it, "
        "and optionally refits LoRA weights from the source model. "
        "Engines are matched by checkpoint name + profile (not filename prefix). "
        "Refitted engines are cached to disk so VRAM eviction doesn't require re-refitting. "
        "Context length is always dynamic — shorter prompts work without rebuilding. "
        "Outputs a MODEL and an info JSON string with engine metadata."
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
        threshold_gb,
        prompt=None,
        unique_id=None,
    ):
        global _refit_cache, _engine_cache
        import comfy.model_management
        import comfy.utils
        from .tensorrt_convert import (
            _derive_model_name,
            _make_profile_desc,
            build_unet_engine,
        )
        from .tensorrt_loader import TrTUnet, _create_model_for_type, _wrap_trt_patcher

        pbar = comfy.utils.ProgressBar(4)

        # Resolve {modelname} placeholder
        derived_model_name = _derive_model_name(prompt, unique_id, "model")
        if "{modelname}" in filename_prefix:
            modelname = derived_model_name or "model"
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
        engine_path = _find_existing_engine(profile_desc, derived_model_name)

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

            evicted = []
            if disk_management == "max_disk_usage":
                evicted = _fifo_evict_max_usage(
                    auto_dir,
                    int(threshold_gb * 1024**3),
                    estimated_new_bytes=2 * 1024**3,
                )
            elif disk_management == "min_disk_free":
                evicted = _fifo_evict_min_free(
                    auto_dir,
                    int(threshold_gb * 1024**3),
                    estimated_new_bytes=2 * 1024**3,
                )
            if evicted:
                PromptServer.instance.send_sync(
                    "trt_disk_eviction",
                    {
                        "evicted": [
                            {"filename": fn, "size_bytes": sz} for fn, sz in evicted
                        ],
                        "total_freed_bytes": sum(sz for _, sz in evicted),
                    },
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
                    1,
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

            _write_meta_sidecar(
                engine_path,
                model_type,
                derived_model_name or filename_prefix,
                profile_desc,
                refit,
            )

        # Build info JSON
        def _build_info(ep):
            meta = _read_meta_sidecar(ep)
            if "model_name" not in meta and derived_model_name:
                meta["model_name"] = derived_model_name
            info = {
                "engine_path": ep,
                "is_symlink": os.path.islink(ep),
                "engine_size_bytes": os.path.getsize(ep) if os.path.isfile(ep) else 0,
                **meta,
            }
            if disk_management != "disabled":
                auto_models = _collect_auto_models_info(auto_dir)
                info["current_disk_usage_bytes"] = sum(
                    m["disk_size_bytes"] for m in auto_models
                )
                info["auto_models"] = auto_models
            return json.dumps(info, indent=2)

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
            cached_unet = _refit_cache["patcher"].model.diffusion_model
            if cached_unet.engine is not None:
                log.info(
                    "Auto: refit cache hit (patches_uuid unchanged), skipping refit"
                )
                _send_trt_progress("cached")
                return (_refit_cache["patcher"], _build_info(engine_path))
            # Engine evicted from VRAM but patches unchanged — check that
            # the persisted refitted engine still exists on disk.
            if cached_unet.engine_path and os.path.isfile(cached_unet.engine_path):
                log.info(
                    "Auto: refit cache hit (evicted, reloading refitted engine on demand)"
                )
                _send_trt_progress("cached")
                return (_refit_cache["patcher"], _build_info(engine_path))
            log.info(
                "Auto: refit cache invalid (persisted engine missing), will re-refit"
            )
            _refit_cache["patcher"] = None

        if (
            not has_patches
            and _engine_cache["engine_path"] == engine_path
            and _engine_cache["patcher"] is not None
        ):
            cached_unet = _engine_cache["patcher"].model.diffusion_model
            if cached_unet.engine is not None:
                log.info("Auto: engine cache hit (no refit needed)")
                _send_trt_progress("cached")
                return (_engine_cache["patcher"], _build_info(engine_path))
            log.info("Auto: engine cache stale (engine evicted from VRAM), will reload")
            _engine_cache["patcher"] = None

        # Step 3: Load engine
        pbar.update_absolute(2, 4)
        _send_trt_progress("loading")
        import torch

        # Free VRAM before probe — TrTUnet.__init__ deserializes the engine
        # to read device_memory_size, which needs the full engine in GPU memory.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        torch.cuda.empty_cache()
        unet = TrTUnet(engine_path)
        model_shell = _create_model_for_type(model_type, unet)
        patcher = _wrap_trt_patcher(model_shell, unet)

        # Step 4: Refit if needed
        pbar.update_absolute(3, 4)
        if has_patches:
            log.info("Auto: refitting with LoRA patches...")
            _send_trt_progress("refitting")

            # Refit in-place — preserves engine_path so ON_LOAD can reload from disk
            unet._load()
            _do_refit(unet.engine, engine_path, model, model_type)

            # Persist refitted engine so ON_LOAD can reload it after VRAM eviction
            # without needing to re-refit (saves ~13s per prompt for SDXL)
            cached_path = _refit_cache_path(engine_path)
            try:
                data = unet.engine.serialize()
                with open(cached_path, "wb") as f:
                    f.write(data)
                del data
                unet.engine_path = cached_path
                log.info("Auto: persisted refitted engine to %s", cached_path)
            except Exception as e:
                log.warning("Auto: failed to persist refitted engine: %s", e)

            _refit_cache["patches_uuid"] = current_uuid
            _refit_cache["engine_path"] = engine_path
            _refit_cache["patcher"] = patcher
        else:
            _engine_cache["engine_path"] = engine_path
            _engine_cache["patcher"] = patcher

        pbar.update_absolute(4, 4)
        _send_trt_progress("done")
        return (patcher, _build_info(engine_path))
