from .nodes import QwenGGUFInference, QwenGGUFParameters


NODE_CLASS_MAPPINGS = {
    "QwenGGUFInference": QwenGGUFInference,
    "QwenGGUFParameters": QwenGGUFParameters,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenGGUFInference": "Qwen 3.5+ GGUF Inference",
    "QwenGGUFParameters": "Qwen GGUF Parameters",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
