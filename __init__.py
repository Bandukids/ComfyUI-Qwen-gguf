from .nodes import QwenGGUFInference, QwenGGUFInferenceAdvanced


NODE_CLASS_MAPPINGS = {
    "QwenGGUFInference": QwenGGUFInference,
    "QwenGGUFInferenceAdvanced": QwenGGUFInferenceAdvanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenGGUFInference": "Qwen 3.5+ GGUF Inference",
    "QwenGGUFInferenceAdvanced": "Qwen 3.5+ GGUF Inference (Advanced)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
