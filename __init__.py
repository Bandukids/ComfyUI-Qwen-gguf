from .nodes import QwenGGUFInference


NODE_CLASS_MAPPINGS = {"QwenGGUFInference": QwenGGUFInference}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenGGUFInference": "Qwen 3.5+ GGUF Inference"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
