# Put this in the custom_nodes folder, put your tensorrt engine files in ComfyUI/models/tensorrt/ (you will have to create the directory)

import torch
import os

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.supported_models
import folder_paths
import logging

trt_logger = logging.getLogger("comfyui_tensorrt")

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

import tensorrt as trt

trt.init_libnvinfer_plugins(None, "")

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)


# Is there a function that already exists for this?
def trt_datatype_to_torch(datatype):
    if datatype == trt.float16:
        return torch.float16
    elif datatype == trt.float32:
        return torch.float32
    elif datatype == trt.int32:
        return torch.int32
    elif datatype == trt.bfloat16:
        return torch.bfloat16


class TrTUnet:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.dtype = torch.float16

        # Probe engine to get memory sizes, then free it immediately.
        # device_memory_size = execution context scratch memory (small)
        # engine file size ≈ GPU weight memory (large, loaded 1:1 to VRAM)
        # We report both so ComfyUI knows the true VRAM cost.
        with open(engine_path, "rb") as f:
            data = f.read()
            engine = runtime.deserialize_cuda_engine(data)
        self.context_memory_size = engine.device_memory_size
        self.engine_weight_size = len(data)
        self.total_vram_size = self.engine_weight_size + self.context_memory_size
        del engine, data
        comfy.model_management.soft_empty_cache()
        trt_logger.info(
            f"TRT UNet probed: {engine_path} "
            f"(weights: {self.engine_weight_size / (1024 * 1024):.0f} MB, "
            f"context: {self.context_memory_size / (1024 * 1024):.0f} MB, "
            f"total: {self.total_vram_size / (1024 * 1024):.0f} MB)"
        )

    def _load(self):
        """Deserialize engine and create execution context."""
        if self.engine is not None:
            return
        trt_logger.info(f"TRT UNet loading engine: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {self.engine_path}\n"
                "The engine file may be corrupt or built with an incompatible TensorRT version. "
                "Try rebuilding the engine."
            )
        self.context = self.engine.create_execution_context()

    def _unload(self):
        """Release engine and execution context VRAM."""
        if self.engine is None:
            return
        trt_logger.info("TRT UNet unloading engine")
        del self.context
        del self.engine
        self.context = None
        self.engine = None

    @classmethod
    def from_engine(cls, engine):
        """Create TrTUnet from an already-deserialized (e.g. refitted) engine."""
        obj = cls.__new__(cls)
        obj.engine_path = None
        obj.engine = engine
        obj.context = engine.create_execution_context()
        obj.dtype = torch.float16
        obj.context_memory_size = engine.device_memory_size
        # Weight memory is already allocated in the engine; use
        # device_memory_size as a conservative total estimate since
        # we can't easily measure serialized size of an in-memory engine.
        obj.engine_weight_size = 0
        obj.total_vram_size = obj.context_memory_size
        return obj

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
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
        opt_batch = dims[1][0]
        max_batch = dims[2][0]

        # Split batch if our batch is bigger than the max batch size the trt engine supports
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

        stream = torch.cuda.default_stream(x.device)
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


class TensorRTLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("tensorrt"),),
                "model_type": (
                    [
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
                    ],
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "TensorRT"

    def load_unet(self, unet_name, model_type):
        unet_path = folder_paths.get_full_path("tensorrt", unet_name)
        if not os.path.isfile(unet_path):
            raise FileNotFoundError(f"File {unet_path} does not exist")
        unet = TrTUnet(unet_path)
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
            unet.dtype = torch.bfloat16  # TODO: autodetect
        elif model_type == "flux_schnell":
            conf = comfy.supported_models.FluxSchnell({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
            unet.dtype = torch.bfloat16  # TODO: autodetect
        model.diffusion_model = unet
        model.memory_required = (
            lambda *args, **kwargs: 0
        )  # always pass inputs batched up as much as possible, our TRT code will handle batch splitting

        # Report the true VRAM cost (weights + context) so ComfyUI makes
        # informed eviction decisions. Previously we only reported context
        # memory (~50 MB), causing ComfyUI to think it could freely evict
        # and reload the 5 GB engine between XY plot iterations.
        trt_memory = unet.total_vram_size
        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
            size=trt_memory,
        )

        def _on_load(p, _device_to, _lowvram, _force_patch, _full_load):
            torch.cuda.empty_cache()  # release PyTorch reserved blocks for TRT
            p.model.diffusion_model._load()
            p.model.model_loaded_weight_memory = trt_memory

        def _on_detach(p, _unpatch_all):
            # Don't unload the engine on detach — keep it hot in VRAM.
            # TRT engine deserialization is very slow (~8s for a 5 GiB engine),
            # and ComfyUI detaches/reattaches the same model between XY plot
            # iterations (for VAE decode). Keeping the engine loaded makes
            # reattach instant. The engine VRAM is freed when the patcher is
            # garbage collected (i.e. when a different model is loaded).
            pass

        patcher.add_callback(comfy.patcher_extension.CallbacksMP.ON_LOAD, _on_load)
        patcher.add_callback(comfy.patcher_extension.CallbacksMP.ON_DETACH, _on_detach)

        trt_logger.info(
            f"TRT UNet registered with ComfyUI memory manager "
            f"({trt_memory / (1024 * 1024):.0f} MB)"
        )

        return (patcher,)


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
        torch.cuda.empty_cache()
        p.model.diffusion_model._load()
        p.model.model_loaded_weight_memory = trt_memory

    def _on_detach(p, _unpatch_all):
        pass

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
                "unet_name": (folder_paths.get_filename_list("tensorrt"),),
                "model_type": (MODEL_TYPE_LIST,),
                "source_model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_and_refit"
    CATEGORY = "TensorRT"
    DESCRIPTION = (
        "Load a refit-enabled TRT engine and update its weights from a source "
        "model (e.g. checkpoint + LoRA). The engine must have been built with "
        "enable_refit=True. This is much faster than rebuilding the engine."
    )

    def load_and_refit(self, unet_name, model_type, source_model):
        unet_path = folder_paths.get_full_path("tensorrt", unet_name)
        if not os.path.isfile(unet_path):
            raise FileNotFoundError(f"File {unet_path} does not exist")

        # --- Step 1: Extract LoRA-patched weights from source model ---
        trt_logger.info("Refit: loading source model to extract weights...")
        comfy.model_management.load_models_gpu(
            [source_model], force_patch_weights=True, force_full_load=True
        )
        src_sd = source_model.model.diffusion_model.state_dict()
        # Copy to CPU numpy immediately so we can free GPU for the TRT engine.
        # Use float16 to match the ONNX export precision.
        weight_dtype = torch.float16
        if model_type in ("flux_dev", "flux_schnell"):
            weight_dtype = torch.bfloat16
        cpu_weights = {}
        for k, v in src_sd.items():
            cpu_weights[k] = v.to(dtype=weight_dtype).cpu().numpy()
        del src_sd
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        # --- Step 2: Deserialize the TRT engine ---
        trt_logger.info(f"Refit: deserializing engine: {unet_path}")
        torch.cuda.empty_cache()
        with open(unet_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {unet_path}\n"
                "The engine file may be corrupt or built with an incompatible "
                "TensorRT version."
            )

        # --- Step 3: Refit weights ---
        trt_logger.info("Refit: applying weights to engine...")
        refitter = trt.Refitter(engine, logger)

        trt_weight_names = set(refitter.get_all_weights())
        if not trt_weight_names:
            del refitter, engine
            raise RuntimeError(
                "Engine has no refittable weights. Was it built with enable_refit=True?"
            )

        matched = 0
        missing_in_trt = 0
        for trt_name in trt_weight_names:
            # TRT weight names have "unet." prefix from the ONNX export wrapper
            if trt_name.startswith("unet."):
                pytorch_key = trt_name[len("unet.") :]
            else:
                pytorch_key = trt_name

            if pytorch_key not in cpu_weights:
                trt_logger.debug(
                    f"Refit: TRT weight '{trt_name}' has no match in source "
                    f"model (may be fused/optimized) — skipping"
                )
                missing_in_trt += 1
                continue

            refitter.set_named_weights(trt_name, trt.Weights(cpu_weights[pytorch_key]))
            matched += 1

        missing = refitter.get_missing_weights()
        if missing:
            trt_logger.warning(
                f"Refit: {len(missing)} weights still missing after mapping: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        trt_logger.info(
            f"Refit: {matched} weights mapped, "
            f"{missing_in_trt} TRT weights without source match, "
            f"{len(missing)} still missing"
        )

        success = refitter.refit_cuda_engine()
        del refitter, cpu_weights
        if not success:
            del engine
            raise RuntimeError(
                "TensorRT engine refit failed. Check the log for details."
            )
        trt_logger.info("Refit: engine weights updated successfully")

        # --- Step 4: Create model patcher ---
        unet = TrTUnet.from_engine(engine)
        model = _create_model_for_type(model_type, unet)
        patcher = _wrap_trt_patcher(model, unet)
        return (patcher,)


NODE_CLASS_MAPPINGS = {
    "TensorRTLoader": TensorRTLoader,
    "TensorRTRefitLoader": TensorRTRefitLoader,
}
