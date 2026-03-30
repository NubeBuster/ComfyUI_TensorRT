from .tensorrt_convert import DYNAMIC_TRT_MODEL_CONVERSION
from .tensorrt_convert import DYNAMIC_VAE_TRT_CONVERSION
from .tensorrt_convert import STATIC_TRT_MODEL_CONVERSION
from .tensorrt_convert import STATIC_VAE_TRT_CONVERSION
from .tensorrt_loader import TensorRTLoader
from .tensorrt_loader import TensorRTRefitLoader
from .tensorrt_loader import TensorRTVAELoader
from .tensorrt_loader import TrTUnet

NODE_CLASS_MAPPINGS = {
    "DYNAMIC_TRT_MODEL_CONVERSION": DYNAMIC_TRT_MODEL_CONVERSION,
    "STATIC_TRT_MODEL_CONVERSION": STATIC_TRT_MODEL_CONVERSION,
    "DYNAMIC_VAE_TRT_CONVERSION": DYNAMIC_VAE_TRT_CONVERSION,
    "STATIC_VAE_TRT_CONVERSION": STATIC_VAE_TRT_CONVERSION,
    "TensorRTLoader": TensorRTLoader,
    "TensorRTRefitLoader": TensorRTRefitLoader,
    "TensorRTVAELoader": TensorRTVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DYNAMIC_TRT_MODEL_CONVERSION": "DYNAMIC TRT MODEL CONVERSION",
    "STATIC_TRT_MODEL_CONVERSION": "STATIC TRT MODEL CONVERSION",
    "DYNAMIC_VAE_TRT_CONVERSION": "DYNAMIC VAE TRT CONVERSION",
    "STATIC_VAE_TRT_CONVERSION": "STATIC VAE TRT CONVERSION",
    "TensorRTLoader": "TensorRT Loader",
    "TensorRTRefitLoader": "TensorRT Refit Loader",
    "TensorRTVAELoader": "TensorRT VAE Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
