# Put this in the custom_nodes folder, put your tensorrt engine files in ComfyUI/models/tensorrt/ (you will have to create the directory)

import json
import re
import torch
import os

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.supported_models
import comfy.utils
import folder_paths
import logging

trt_logger = logging.getLogger("comfyui_tensorrt")

# Wrap unload_all_models to signal that ON_DETACH should actually free engines.
# Without this, ON_DETACH is a no-op (keeps engines hot for XY plot eviction).
_trt_force_unload = False
_original_unload_all = comfy.model_management.unload_all_models


def _wrapped_unload_all():
    global _trt_force_unload
    _trt_force_unload = True
    try:
        _original_unload_all()
    finally:
        _trt_force_unload = False


comfy.model_management.unload_all_models = _wrapped_unload_all


def _vram_snapshot(label=""):
    """Log current VRAM usage for debugging model lifecycle."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        free, total = torch.cuda.mem_get_info()
        free_mb = free / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        trt_logger.info(
            f"[VRAM {label}] allocated={alloc:.0f} MB, reserved={reserved:.0f} MB, "
            f"free={free_mb:.0f}/{total_mb:.0f} MB"
        )
    except Exception as e:
        trt_logger.debug(f"[VRAM {label}] snapshot failed: {e}")


if "tensorrt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt"][0].append(
        os.path.join(folder_paths.models_dir, "tensorrt")
    )
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
else:
    folder_paths.folder_names_and_paths["tensorrt"] = (
        [os.path.join(folder_paths.models_dir, "tensorrt")],
        {".engine"},
    )

# Register auto-managed engine directory
_AUTO_ENGINE_DIR = os.path.join(folder_paths.models_dir, "tensorrt", "auto")
os.makedirs(_AUTO_ENGINE_DIR, exist_ok=True)
if "tensorrt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt"][0].append(_AUTO_ENGINE_DIR)

import tensorrt as trt

trt.init_libnvinfer_plugins(None, "")

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)

# Multi-slot refit cache: hash-based, supports LoRA cycling (A→B→A)
from . import refit_cache as _rc


def trt_datatype_to_torch(datatype):
    if datatype == trt.float16:
        return torch.float16
    elif datatype == trt.float32:
        return torch.float32
    elif datatype == trt.int32:
        return torch.int32
    elif datatype == trt.bfloat16:
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported TRT dtype: {datatype}")


class TrTEngine:
    """Base TRT engine wrapper with probe/load/unload lifecycle."""

    def __init__(self, engine_path, label="TRT"):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self._label = label

        # Probe engine to get memory sizes, then free it immediately.
        # device_memory_size = execution context scratch memory (small)
        # engine file size ≈ GPU weight memory (large, loaded 1:1 to VRAM)
        # We report both so ComfyUI knows the true VRAM cost.
        _vram_snapshot(f"{label} pre-probe")
        with open(engine_path, "rb") as f:
            data = f.read()
            engine = runtime.deserialize_cuda_engine(data)
        if engine is None:
            _vram_snapshot(f"{label} probe FAILED (deserialize returned None)")
            raise RuntimeError(
                f"TRT engine probe failed — deserialize returned None for: "
                f"{engine_path}\n"
                f"File size: {len(data) / (1024 * 1024):.0f} MB. "
                f"Likely CUDA OOM during deserialization. "
                f"Check [VRAM] log lines above for memory state."
            )
        self.context_memory_size = engine.device_memory_size
        self.engine_weight_size = len(data)
        self.total_vram_size = self.engine_weight_size + self.context_memory_size
        del engine, data
        comfy.model_management.soft_empty_cache()
        _vram_snapshot(f"{label} post-probe")
        trt_logger.info(
            f"{label} probed: {engine_path} "
            f"(weights: {self.engine_weight_size / (1024 * 1024):.0f} MB, "
            f"context: {self.context_memory_size / (1024 * 1024):.0f} MB)"
        )

    def _load(self):
        """Deserialize engine and create execution context."""
        if self.engine is not None:
            trt_logger.debug(
                f"[lifecycle] {self._label} _load() skipped — engine already loaded "
                f"(path={self.engine_path})"
            )
            return
        if self.engine_path is None:
            raise RuntimeError(
                "Cannot reload a refitted TRT engine — it exists only in memory. "
                "Re-run the refit node to recreate it."
            )
        _vram_snapshot(f"{self._label} pre-load")
        trt_logger.info(f"[lifecycle] {self._label} loading engine: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            _vram_snapshot(f"{self._label} load FAILED")
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {self.engine_path}\n"
                "The engine file may be corrupt or built with an incompatible "
                "TensorRT version. Try rebuilding the engine."
            )
        self.context = self.engine.create_execution_context()
        _vram_snapshot(f"{self._label} post-load")

    def _unload(self):
        """Release engine and execution context VRAM."""
        if self.engine is None:
            trt_logger.debug(
                f"[lifecycle] {self._label} _unload() skipped — already unloaded"
            )
            return
        _vram_snapshot(f"{self._label} pre-unload")
        trt_logger.info(
            f"[lifecycle] {self._label} unloading engine (path={self.engine_path})"
        )
        del self.context
        del self.engine
        self.context = None
        self.engine = None
        import torch

        torch.cuda.empty_cache()
        _vram_snapshot(f"{self._label} post-unload")

    @classmethod
    def from_engine(cls, engine):
        """Create from an already-deserialized engine (e.g. after refit)."""
        obj = cls.__new__(cls)
        obj._label = "TRT"
        obj.engine_path = None
        obj.engine = engine
        obj.context = engine.create_execution_context()
        obj.context_memory_size = engine.device_memory_size
        # Weight memory is already allocated in the engine; use
        # device_memory_size as a conservative total estimate since
        # we can't easily measure serialized size of an in-memory engine.
        obj.engine_weight_size = 0
        obj.total_vram_size = obj.context_memory_size
        return obj


class TrTUnet(TrTEngine):
    def __init__(self, engine_path):
        super().__init__(engine_path, label="TRT UNet")
        self.dtype = torch.float16

    @classmethod
    def from_engine(cls, engine):
        """Create TrTUnet from an already-deserialized (e.g. refitted) engine."""
        obj = super().from_engine(engine)
        obj.dtype = torch.float16
        return obj

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            # Validate shape against engine profile
            try:
                profile_min, profile_opt, profile_max = (
                    self.engine.get_tensor_profile_shape(k, 0)
                )
                for d, (lo, hi, actual) in enumerate(
                    zip(profile_min, profile_max, shape)
                ):
                    if actual < lo or actual > hi:
                        raise ValueError(
                            f"TRT input '{k}' dimension {d}: got {actual}, "
                            f"engine expects [{lo}..{hi}]. "
                            f"Full input shape: {list(shape)}, "
                            f"engine range: {list(profile_min)}..{list(profile_max)}. "
                            f"Rebuild the engine with matching dimensions."
                        )
            except RuntimeError:
                pass  # Not all tensors have profile shapes (e.g. scalar inputs)
            self.context.set_input_shape(k, shape)

    def __call__(
        self,
        x,
        timesteps,
        context,
        y=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        # Ensure engine is loaded (ON_LOAD callback should have done this,
        # but guard against edge cases)
        self._load()

        model_inputs = {"x": x, "timesteps": timesteps, "context": context}

        if y is not None:
            model_inputs["y"] = y

        for i in range(len(model_inputs), self.engine.num_io_tensors - 1):
            name = self.engine.get_tensor_name(i)
            model_inputs[name] = kwargs[name]

        batch_size = x.shape[0]
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        max_batch = dims[2][0]

        # Split batch if our batch is bigger than the max batch size the trt engine supports
        curr_split_batch = 1
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs, curr_split_batch)

        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            model_inputs_converted[k] = model_inputs[k].to(
                dtype=trt_datatype_to_torch(data_type)
            )

        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)

        # for dynamic profile case where the dynamic params are -1
        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(
            out_shape,
            device=x.device,
            dtype=trt_datatype_to_torch(
                self.engine.get_tensor_dtype(output_binding_name)
            ),
        )
        model_inputs_converted[output_binding_name] = out

        stream = torch.cuda.current_stream(x.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                x = model_inputs_converted[k]
                self.context.set_tensor_address(
                    k, x[(x.shape[0] // curr_split_batch) * i :].data_ptr()
                )
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        # stream.synchronize() #don't need to sync stream since it's the default torch one
        return out

    def load_state_dict(self, sd, strict=False):
        pass

    def state_dict(self):
        return {}


MODEL_TYPE_LIST = [
    "sdxl_base",
    "sdxl_inpaint",
    "sdxl_refiner",
    "sd1.x",
    "sd2.x-768v",
    "svd",
    "sd3",
    "auraflow",
    "flux_dev",
    "flux_schnell",
]


class TensorRTLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (
                    _list_unet_engines(),
                    {
                        "tooltip": "TensorRT engine file to load. Only UNet engines are shown (VAE engines are filtered out)."
                    },
                ),
                "model_type": (
                    MODEL_TYPE_LIST,
                    {
                        "tooltip": "Architecture of the model the engine was built from. Must match exactly."
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "TensorRT"
    DESCRIPTION = (
        "Load a pre-built TensorRT engine as a UNet model. "
        "The engine is managed by ComfyUI's VRAM system — it loads to GPU on demand "
        "and unloads when other models need space. "
        "Input shapes are validated before inference — mismatches error immediately "
        "instead of producing corrupt output."
    )

    def load_unet(self, unet_name, model_type):
        unet_path = folder_paths.get_full_path("tensorrt", unet_name)
        if not os.path.isfile(unet_path):
            raise FileNotFoundError(f"File {unet_path} does not exist")
        unet = TrTUnet(unet_path)
        model = _create_model_for_type(model_type, unet)
        patcher = _wrap_trt_patcher(model, unet)
        return (patcher,)


def _create_model_for_type(model_type, unet):
    """Create a ComfyUI model shell for the given model_type and attach unet."""
    if model_type in ("sdxl_base", "sdxl_inpaint"):
        conf = comfy.supported_models.SDXL({"adm_in_channels": 2816})
        conf.unet_config["disable_unet_model_creation"] = True
        model = comfy.model_base.SDXL(conf)
        if model_type == "sdxl_inpaint":
            model.set_inpaint()
    elif model_type == "sdxl_refiner":
        conf = comfy.supported_models.SDXLRefiner({"adm_in_channels": 2560})
        conf.unet_config["disable_unet_model_creation"] = True
        model = comfy.model_base.SDXLRefiner(conf)
    elif model_type == "sd1.x":
        conf = comfy.supported_models.SD15({})
        conf.unet_config["disable_unet_model_creation"] = True
        model = comfy.model_base.BaseModel(conf)
    elif model_type == "sd2.x-768v":
        conf = comfy.supported_models.SD20({})
        conf.unet_config["disable_unet_model_creation"] = True
        model = comfy.model_base.BaseModel(
            conf, model_type=comfy.model_base.ModelType.V_PREDICTION
        )
    elif model_type == "svd":
        conf = comfy.supported_models.SVD_img2vid({})
        conf.unet_config["disable_unet_model_creation"] = True
        model = conf.get_model({})
    elif model_type == "sd3":
        conf = comfy.supported_models.SD3({})
        conf.unet_config["disable_unet_model_creation"] = True
        model = conf.get_model({})
    elif model_type == "auraflow":
        conf = comfy.supported_models.AuraFlow({})
        conf.unet_config["disable_unet_model_creation"] = True
        model = conf.get_model({})
    elif model_type == "flux_dev":
        conf = comfy.supported_models.Flux({})
        conf.unet_config["disable_unet_model_creation"] = True
        model = conf.get_model({})
        unet.dtype = torch.bfloat16
    elif model_type == "flux_schnell":
        conf = comfy.supported_models.FluxSchnell({})
        conf.unet_config["disable_unet_model_creation"] = True
        model = conf.get_model({})
        unet.dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.diffusion_model = unet
    model.memory_required = lambda *args, **kwargs: 0
    return model


def _wrap_trt_patcher(model, unet):
    """Wrap a TRT model in a ModelPatcher with load/unload callbacks."""
    trt_memory = unet.total_vram_size
    patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
        size=trt_memory,
    )

    def _on_load(p, _device_to, _lowvram, _force_patch, _full_load):
        engine_loaded = p.model.diffusion_model.engine is not None
        trt_logger.info(
            f"[lifecycle] UNet ON_LOAD called (engine_already_loaded={engine_loaded}, "
            f"path={p.model.diffusion_model.engine_path})"
        )
        _vram_snapshot("UNet ON_LOAD pre")
        torch.cuda.empty_cache()
        p.model.diffusion_model._load()
        p.model.model_loaded_weight_memory = trt_memory
        _vram_snapshot("UNet ON_LOAD post")

    def _on_detach(p, _unpatch_all):
        if not _trt_force_unload:
            trt_logger.info("[lifecycle] UNet ON_DETACH — keeping engine hot")
            if trt_logger.isEnabledFor(logging.DEBUG):
                import traceback

                trt_logger.debug(
                    "  caller: %s", "".join(traceback.format_stack()[-4:-1]).strip()
                )
            return
        trt_logger.info("[lifecycle] UNet ON_DETACH — unloading engine (force)")
        if trt_logger.isEnabledFor(logging.DEBUG):
            import traceback

            trt_logger.debug(
                "  caller: %s", "".join(traceback.format_stack()[-4:-1]).strip()
            )
        _vram_snapshot("UNet ON_DETACH pre")
        p.model.diffusion_model._unload()
        _vram_snapshot("UNet ON_DETACH post")

    patcher.add_callback(comfy.patcher_extension.CallbacksMP.ON_LOAD, _on_load)
    patcher.add_callback(comfy.patcher_extension.CallbacksMP.ON_DETACH, _on_detach)

    trt_logger.info(
        f"TRT UNet registered with ComfyUI memory manager "
        f"({trt_memory / (1024 * 1024):.0f} MB)"
    )
    return patcher


class TensorRTRefitLoader:
    """Load a refit-enabled TRT engine and update its weights from a source model.

    This allows swapping LoRA weights into a pre-built TRT engine in seconds
    instead of rebuilding the engine from scratch (minutes).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (
                    _list_refit_engines(),
                    {
                        "tooltip": "Refit-enabled TensorRT engine file. Must have been built with enable_refit=True."
                    },
                ),
                "model_type": (
                    MODEL_TYPE_LIST,
                    {
                        "tooltip": "Architecture of the model the engine was built from. Must match exactly."
                    },
                ),
                "source_model": (
                    "MODEL",
                    {
                        "tooltip": "Source model with LoRA/weights applied. Its weights are refitted into the TRT engine. "
                        "Results are cached by a hash of the LoRA patches — cycling between LoRA configs (A→B→A) skips refitting on revisits."
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_and_refit"
    CATEGORY = "TensorRT"
    DESCRIPTION = (
        "Load a refit-enabled TRT engine and update its weights from a source "
        "model (e.g. checkpoint + LoRA). Much faster than rebuilding (~13s vs 5-10 min). "
        "Refitted engines are cached to disk — repeated runs with the same LoRAs "
        "skip refitting entirely, even after VRAM eviction. "
        "Cache is invalidated automatically when LoRA patches change."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def load_and_refit(self, unet_name, model_type, source_model):
        if unet_name == _NO_REFIT_ENGINES:
            raise ValueError(
                "No refit-enabled TRT engines found. Build an engine with "
                "enable_refit=True first."
            )
        unet_path = folder_paths.get_full_path("tensorrt", unet_name)
        if not os.path.isfile(unet_path):
            raise FileNotFoundError(f"File {unet_path} does not exist")

        # Cache check: hash-based multi-slot cache for LoRA cycling
        lora_hash = _rc.compute_patches_hash(source_model)
        disk_path = None
        if lora_hash is not None:
            cached_patcher = _rc.mem_lookup(unet_path, lora_hash)
            if cached_patcher is not None:
                return (cached_patcher,)

            # Memory miss — check disk cache
            disk_path = _rc.disk_lookup(unet_path, lora_hash)
            if disk_path is not None:
                trt_logger.info("Refit: disk cache hit, loading hash=%s", lora_hash)
            else:
                trt_logger.info("Refit: cache MISS (hash=%s not found)", lora_hash)

        pbar = comfy.utils.ProgressBar(4)

        # Disk cache hit — load pre-refitted engine, skip full refit
        if disk_path is not None:
            pbar.update_absolute(1, 4)
            _rc.mem_release_all_engines()
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            torch.cuda.empty_cache()
            unet = TrTUnet(disk_path)
            model = _create_model_for_type(model_type, unet)
            patcher = _wrap_trt_patcher(model, unet)
            _rc.mem_store(unet_path, lora_hash, patcher)
            pbar.update_absolute(4, 4)
            trt_logger.info("Refit: loaded from disk cache hash=%s", lora_hash)
            return (patcher,)

        # --- Step 1: Extract LoRA-patched weights from source model ---
        pbar.update_absolute(0, 4)
        trt_logger.info("Refit: loading source model to extract weights...")
        comfy.model_management.load_models_gpu([source_model])

        # Use patch_weight_to_device to compute patched weights without
        # force_patch_weights (incompatible with ModelPatcherDynamic).
        weight_dtype = torch.float16
        if model_type in ("flux_dev", "flux_schnell"):
            weight_dtype = torch.bfloat16

        patcher_type = type(source_model).__name__
        trt_logger.info(
            f"Refit: source patcher type={patcher_type}, "
            f"patches={len(source_model.patches)}, "
            f"weight_dtype={weight_dtype}"
        )

        # Get base weights first for delta comparison
        base_sd = source_model.model.diffusion_model.state_dict()
        trt_logger.info(f"Refit: base state_dict has {len(base_sd)} keys")

        diffusion_prefix = "diffusion_model."
        cpu_weights = {}
        patched_count = 0
        skipped_none = 0
        skipped_noprefix = 0
        delta_nonzero = 0
        for key in list(source_model.patches.keys()):
            if not key.startswith(diffusion_prefix):
                skipped_noprefix += 1
                continue
            w = source_model.patch_weight_to_device(key, return_weight=True)
            if w is None:
                skipped_none += 1
                trt_logger.warning(
                    f"Refit: patch_weight_to_device returned None for {key}"
                )
                continue
            short_key = key[len(diffusion_prefix) :]
            # Check if patched weight differs from base
            if short_key in base_sd:
                base_w = base_sd[short_key].to(dtype=w.dtype, device=w.device)
                diff = (w - base_w).abs().max().item()
                if diff > 1e-6:
                    delta_nonzero += 1
                    if delta_nonzero <= 3:
                        trt_logger.info(
                            f"Refit: LoRA delta sample: {short_key} "
                            f"max_diff={diff:.6f}, shape={list(w.shape)}"
                        )
            cpu_weights[short_key] = w.to(dtype=weight_dtype).cpu().numpy()
            patched_count += 1

        trt_logger.info(
            f"Refit: {patched_count} patched weights extracted, "
            f"{delta_nonzero} differ from base, "
            f"{skipped_none} returned None, "
            f"{skipped_noprefix} skipped (non-diffusion)"
        )

        # Track which keys had LoRA patches applied
        patched_keys = set(cpu_weights.keys())

        # Fill in base (unpatched) weights for keys not covered by patches.
        base_count = 0
        for k in list(base_sd.keys()):
            if k not in cpu_weights:
                cpu_weights[k] = base_sd.pop(k).to(dtype=weight_dtype).cpu().numpy()
                base_count += 1
        trt_logger.info(
            f"Refit: {base_count} base weights added (total: {len(cpu_weights)})"
        )
        del base_sd
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        # --- Step 2: Load TRT engine (preserves engine_path for reload) ---
        pbar.update_absolute(1, 4)
        trt_logger.info(f"Refit: loading engine: {unet_path}")
        torch.cuda.empty_cache()
        unet = TrTUnet(unet_path)
        unet._load()

        # --- Step 3: Refit weights in-place ---
        pbar.update_absolute(2, 4)
        trt_logger.info("Refit: applying weights to engine...")
        engine = unet.engine
        refitter = trt.Refitter(engine, logger)

        trt_weight_names = set(refitter.get_all_weights())
        trt_logger.info(f"Refit: TRT has {len(trt_weight_names)} refittable weights")
        if not trt_weight_names:
            del refitter, engine
            raise RuntimeError(
                "Engine has no refittable weights. Was it built with enable_refit=True?"
            )

        # Load ONNX weight name mapping (maps onnx::MatMul_* -> pytorch keys)
        weight_map_path = unet_path.replace(".engine", ".weight_map.json")
        onnx_weight_map = {}
        if os.path.isfile(weight_map_path):
            with open(weight_map_path) as f:
                onnx_weight_map = json.load(f)
            trt_logger.info(
                f"Refit: loaded weight map with {len(onnx_weight_map)} entries"
            )
        else:
            trt_logger.warning(
                "Refit: no .weight_map.json sidecar found — "
                "onnx::* weights will not be mapped to LoRA keys. "
                "Rebuild the engine to generate the mapping."
            )

        import numpy as np

        matched = 0
        matched_patched = 0
        matched_base = 0
        missing_in_trt = 0
        mapped_via_onnx = 0
        for trt_name in trt_weight_names:
            via_onnx = False
            # TRT weight names have "unet." prefix from the ONNX export wrapper
            if trt_name.startswith("unet."):
                pytorch_key = trt_name[len("unet.") :]
            elif trt_name in onnx_weight_map:
                pytorch_key = onnx_weight_map[trt_name]
                via_onnx = True
                mapped_via_onnx += 1
            else:
                pytorch_key = trt_name

            if pytorch_key not in cpu_weights:
                trt_logger.debug(
                    f"Refit: TRT weight '{trt_name}' -> '{pytorch_key}' "
                    f"has no match in source model — skipping"
                )
                missing_in_trt += 1
                continue

            w = cpu_weights[pytorch_key]
            # ONNX MatMul weights need transposition: PyTorch nn.Linear stores
            # [out, in] but ONNX MatMul expects [in, out] as the B matrix.
            if via_onnx and "MatMul" in trt_name and w.ndim == 2:
                w = np.ascontiguousarray(w.T)

            refitter.set_named_weights(trt_name, trt.Weights(w))
            matched += 1
            if pytorch_key in patched_keys:
                matched_patched += 1
            else:
                matched_base += 1

        # Build the set of all pytorch keys reachable via TRT weight mapping
        all_mapped_keys = set()
        for trt_name in trt_weight_names:
            if trt_name.startswith("unet."):
                all_mapped_keys.add(trt_name[len("unet.") :])
            elif trt_name in onnx_weight_map:
                all_mapped_keys.add(onnx_weight_map[trt_name])

        patched_not_mapped = patched_keys - all_mapped_keys
        if patched_not_mapped:
            samples = sorted(patched_not_mapped)[:5]
            trt_logger.warning(
                f"Refit: {len(patched_not_mapped)} LoRA-patched weights NOT "
                f"mapped to any TRT weight: {samples}"
                f"{'...' if len(patched_not_mapped) > 5 else ''}"
            )

        missing = refitter.get_missing_weights()
        if missing:
            trt_logger.warning(
                f"Refit: {len(missing)} weights still missing after mapping: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        trt_logger.info(
            f"Refit: {matched} weights mapped "
            f"({matched_patched} LoRA-patched, {matched_base} base, "
            f"{mapped_via_onnx} via ONNX map), "
            f"{missing_in_trt} TRT weights without source match, "
            f"{len(missing)} still missing"
        )

        success = refitter.refit_cuda_engine()
        del refitter, cpu_weights
        if not success:
            raise RuntimeError(
                "TensorRT engine refit failed. Check the log for details."
            )
        trt_logger.info("Refit: engine weights updated successfully")

        # Persist refitted engine with hash-based filename
        if lora_hash is not None:
            cached_path = _rc.disk_cache_path(unet_path, lora_hash)
        else:
            cache_dir = os.path.join(os.path.dirname(unet_path), ".refit_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cached_path = os.path.join(cache_dir, os.path.basename(unet_path))
        try:
            data = unet.engine.serialize()
            with open(cached_path, "wb") as f:
                f.write(data)
            del data
            unet.engine_path = cached_path
            trt_logger.info("Refit: persisted refitted engine to %s", cached_path)
        except Exception as e:
            trt_logger.warning("Refit: failed to persist refitted engine: %s", e)

        # --- Step 4: Create model patcher (engine_path preserved for reload) ---
        pbar.update_absolute(3, 4)
        model = _create_model_for_type(model_type, unet)
        patcher = _wrap_trt_patcher(model, unet)

        if lora_hash is not None:
            _rc.mem_store(unet_path, lora_hash, patcher)

        pbar.update_absolute(4, 4)
        return (patcher,)


# --- VAE TRT Loading ---

# Pattern to strip _decode_ or _encode_ and everything after to get a base key
# e.g. "ComfyUI_VAE_STAT_decode_$stat-h-1024-w-1024_00001_.engine"
#   -> base key "ComfyUI_VAE_STAT_$stat-h-1024-w-1024_00001_"
_ENGINE_OP_RE = re.compile(r"_(decode|encode)_")
_REFIT_RE = re.compile(r"_refit_")
_NO_REFIT_ENGINES = "(no refit engines found)"
_NO_VAE_ENGINES = "(no VAE engines found)"
_vae_engine_set_map = {}


def _list_unet_engines():
    """List TRT engine files, excluding VAE engines. Includes refit-enabled engines."""
    return [
        f
        for f in folder_paths.get_filename_list("tensorrt")
        if not _ENGINE_OP_RE.search(f)
    ]


def _list_refit_engines():
    """List refit-enabled TRT engine files (excluding VAE engines)."""
    engines = [
        f
        for f in folder_paths.get_filename_list("tensorrt")
        if _REFIT_RE.search(f) and not _ENGINE_OP_RE.search(f)
    ]
    return engines if engines else [_NO_REFIT_ENGINES]


_VAE_ENGINE_DIR = os.path.join(folder_paths.models_dir, "tensorrt", "vae")


def _list_vae_engine_sets():
    """Discover VAE engine sets by grouping decode/encode pairs by base key.

    Lists the vae/ subdirectory directly (bare filenames, no path prefix).
    Returns a list of display keys for the dropdown. Each key maps to
    a pair of engine filenames (decode + optional encode).
    """
    global _vae_engine_set_map
    _vae_engine_set_map = {}

    if not os.path.exists(_VAE_ENGINE_DIR):
        return [_NO_VAE_ENGINES]

    groups = {}  # base_key -> {"decode": filename, "encode": filename}
    for f in sorted(os.listdir(_VAE_ENGINE_DIR)):
        if not f.endswith(".engine"):
            continue
        stem = f[: -len(".engine")]
        match = _ENGINE_OP_RE.search(stem)
        if not match:
            continue
        operation = match.group(1)
        # Remove _decode or _encode, keep the trailing _
        base_key = stem[: match.start()] + stem[match.end() - 1 :]
        if base_key not in groups:
            groups[base_key] = {}
        groups[base_key][operation] = f

    if not groups:
        return [_NO_VAE_ENGINES]

    keys = []
    for base_key in sorted(groups):
        group = groups[base_key]
        if "decode" not in group:
            continue  # Need at least decode
        _vae_engine_set_map[base_key] = group
        keys.append(base_key)

    return keys if keys else [_NO_VAE_ENGINES]


class TrTVae(TrTEngine):
    """TRT engine wrapper for VAE operations."""

    def __init__(self, engine_path):
        super().__init__(engine_path, label="TRT VAE")

    def __call__(self, input_tensor, output_shape):
        """Run single input -> single output inference.

        Mirrors the allocate_buffers + infer pattern from Engine (trt_engine.py):
        pre-allocate contiguous I/O buffers, copy input, execute, return output.
        """
        self._load()

        # Allocate contiguous I/O buffers (same pattern as Engine.allocate_buffers)
        in_dtype = trt_datatype_to_torch(self.engine.get_tensor_dtype("input"))
        in_buf = torch.empty(
            list(input_tensor.shape), dtype=in_dtype, device=input_tensor.device
        )
        self.context.set_input_shape("input", list(input_tensor.shape))

        out_dtype = trt_datatype_to_torch(self.engine.get_tensor_dtype("output"))
        out_buf = torch.empty(output_shape, dtype=out_dtype, device=input_tensor.device)

        # Copy input into contiguous buffer, set addresses, execute
        in_buf.copy_(input_tensor)
        self.context.set_tensor_address("input", in_buf.data_ptr())
        self.context.set_tensor_address("output", out_buf.data_ptr())

        stream = torch.cuda.current_stream(input_tensor.device)
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT VAE inference failed")
        torch.cuda.synchronize()
        return out_buf


class TrtVAE:
    """ComfyUI VAE-compatible wrapper backed by TensorRT engines.

    Supports AutoencoderKL (SD1.5/SDXL). Outputs from TensorRTVAELoader
    can be used with any node that accepts VAE (VAEDecode, VAEEncode, etc.).
    """

    def __init__(self, decode_engine_path, encode_engine_path=None):
        self.decode_eng = TrTVae(decode_engine_path) if decode_engine_path else None
        self.encode_eng = TrTVae(encode_engine_path) if encode_engine_path else None

        # Standard ComfyUI VAE interface attributes
        self.output_channels = 3
        self.device = torch.device("cuda")
        self.vae_dtype = torch.float16
        self.output_device = torch.device("cpu")
        self.latent_channels = 4
        self.latent_dim = 2
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.working_dtypes = [torch.float16]
        self.first_stage_model = None
        self.process_input = lambda x: x * 2.0 - 1.0
        self.process_output = lambda x: torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        self.memory_used_encode = lambda shape, dtype: 0
        self.memory_used_decode = lambda shape, dtype: 0

        # VRAM estimate from engine probes (more accurate than file size)
        trt_mem = sum(
            e.total_vram_size for e in [self.decode_eng, self.encode_eng] if e
        )
        outer = torch.nn.Module()
        outer.model_loaded_weight_memory = 0
        self.patcher = comfy.model_patcher.ModelPatcher(
            outer,
            load_device=comfy.model_management.vae_device(),
            offload_device=comfy.model_management.vae_offload_device(),
            size=trt_mem,
        )

        decode_eng = self.decode_eng
        encode_eng = self.encode_eng

        def _on_load(p, _device_to, _lowvram, _force_patch, _full_load):
            dec_loaded = decode_eng.engine is not None if decode_eng else None
            enc_loaded = encode_eng.engine is not None if encode_eng else None
            trt_logger.info(
                f"[lifecycle] VAE ON_LOAD called "
                f"(decode_loaded={dec_loaded}, encode_loaded={enc_loaded})"
            )
            _vram_snapshot("VAE ON_LOAD pre")
            torch.cuda.empty_cache()
            if decode_eng:
                decode_eng._load()
            if encode_eng:
                encode_eng._load()
            p.model.model_loaded_weight_memory = trt_mem
            _vram_snapshot("VAE ON_LOAD post")

        def _on_detach(p, _unpatch_all):
            if not _trt_force_unload:
                trt_logger.info("[lifecycle] VAE ON_DETACH — keeping engines hot")
                if trt_logger.isEnabledFor(logging.DEBUG):
                    import traceback

                    trt_logger.debug(
                        "  caller: %s", "".join(traceback.format_stack()[-4:-1]).strip()
                    )
                return
            trt_logger.info("[lifecycle] VAE ON_DETACH — unloading engines (force)")
            if trt_logger.isEnabledFor(logging.DEBUG):
                import traceback

                trt_logger.debug(
                    "  caller: %s", "".join(traceback.format_stack()[-4:-1]).strip()
                )
            _vram_snapshot("VAE ON_DETACH pre")
            if decode_eng:
                decode_eng._unload()
            if encode_eng:
                encode_eng._unload()
            _vram_snapshot("VAE ON_DETACH post")

        self.patcher.add_callback(comfy.patcher_extension.CallbacksMP.ON_LOAD, _on_load)
        self.patcher.add_callback(
            comfy.patcher_extension.CallbacksMP.ON_DETACH, _on_detach
        )

        trt_logger.info(
            f"TRT VAE registered with ComfyUI memory manager "
            f"({trt_mem / (1024 * 1024):.0f} MB)"
        )

    def throw_exception_if_invalid(self):
        pass

    def decode(self, samples_in, **kwargs):
        if self.decode_eng is None:
            raise ValueError("No TRT decode engine loaded")
        B, C, H, W = samples_in.shape
        comfy.model_management.load_models_gpu([self.patcher])

        scale = self.upscale_ratio
        inp = samples_in.to(dtype=self.vae_dtype, device=self.device)
        out = self.decode_eng(inp, (B, self.output_channels, H * scale, W * scale))
        out = self.process_output(out.float())
        return out.movedim(1, -1).to(self.output_device)  # BCHW -> BHWC

    def encode(self, pixel_samples, **kwargs):
        if self.encode_eng is None:
            raise ValueError("No TRT encode engine loaded")
        x = pixel_samples.movedim(-1, 1)  # BHWC -> BCHW
        x = self.process_input(x)
        B, C, H, W = x.shape
        comfy.model_management.load_models_gpu([self.patcher])

        scale = self.downscale_ratio
        inp = x.to(dtype=self.vae_dtype, device=self.device)
        latent = self.encode_eng(inp, (B, self.latent_channels, H // scale, W // scale))
        return latent.float().to(self.output_device)

    def decode_tiled(self, *args, **kwargs):
        raise NotImplementedError("Tiled decoding not supported with TRT engines")

    def encode_tiled(self, *args, **kwargs):
        raise NotImplementedError("Tiled encoding not supported with TRT engines")

    def vae_encode_crop_pixels(self, pixels):
        h = (pixels.shape[1] // self.downscale_ratio) * self.downscale_ratio
        w = (pixels.shape[2] // self.downscale_ratio) * self.downscale_ratio
        return pixels[:, :h, :w, :]

    def spacial_compression_encode(self):
        return self.downscale_ratio

    def spacial_compression_decode(self):
        return self.upscale_ratio


class TensorRTVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "engine": (
                    _list_vae_engine_sets(),
                    {
                        "tooltip": "VAE engine set to load. Shows matched decode+encode pairs discovered from the engine directory."
                    },
                ),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "TensorRT"
    DESCRIPTION = "Load pre-built VAE TensorRT engines (decode + optional encode). Engines are auto-discovered as matched pairs from the engine directory."

    def load_vae(self, engine):
        if engine == _NO_VAE_ENGINES:
            raise ValueError(
                "No VAE TRT engines found. Build them first with a "
                "VAE TRT Conversion node."
            )

        engine_set = _vae_engine_set_map.get(engine)
        if not engine_set:
            raise ValueError(f"Engine set '{engine}' not found. Refresh the page.")

        dec_file = engine_set["decode"]
        enc_file = engine_set.get("encode")

        dec_path = os.path.join(_VAE_ENGINE_DIR, dec_file)
        if not os.path.isfile(dec_path):
            raise FileNotFoundError(f"Decode engine not found: {dec_path}")

        enc_path = None
        if enc_file:
            enc_path = os.path.join(_VAE_ENGINE_DIR, enc_file)
            if not os.path.isfile(enc_path):
                raise FileNotFoundError(f"Encode engine not found: {enc_path}")

        vae = TrtVAE(dec_path, enc_path)
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "TensorRTLoader": TensorRTLoader,
    "TensorRTRefitLoader": TensorRTRefitLoader,
    "TensorRTVAELoader": TensorRTVAELoader,
}
