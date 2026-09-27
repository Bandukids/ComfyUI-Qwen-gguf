"""Minimal Qwen 3.5+ GGUF image/text inference node for ComfyUI."""

import atexit
import base64
import gc
import io
from pathlib import Path
import threading

import folder_paths
from PIL import Image


MODEL_CATEGORY = "qwen_gguf"
MODEL_DIR = Path(folder_paths.models_dir) / "LLM"
folder_paths.add_model_folder_path(MODEL_CATEGORY, str(MODEL_DIR))

_LOCK = threading.RLock()
_MODEL = None
_CONFIG = None

PROMPT_PRESETS = {
    "None": "",
    "Detailed Description": "Describe the image in detail, including its subjects, setting, actions, and visible text.",
    "Brief Description": "Describe the image briefly and accurately.",
    "Extract Text (OCR)": "Transcribe all readable text in the image. Preserve its original language and line breaks.",
    "Compare Images": "Compare the images in order. Explain their important similarities and differences.",
}

ATTENTION_MODES = {"auto": -1, "disabled": 0, "enabled": 1}


def _gguf_files():
    return [name for name in folder_paths.get_filename_list(MODEL_CATEGORY)
            if name.lower().endswith(".gguf")]


def _model_path(name):
    # A workflow can submit strings directly through the ComfyUI API.
    path = Path(name)
    if not name or path.is_absolute() or path.drive or ".." in path.parts or path.suffix.lower() != ".gguf":
        raise ValueError(f"Invalid GGUF filename: {name!r}")
    full_path = folder_paths.get_full_path(MODEL_CATEGORY, name)
    if full_path is None:
        raise FileNotFoundError(f"GGUF file not found: {name!r}. Put it under {MODEL_DIR}")
    return str(Path(full_path).resolve())


def _close_model():
    global _MODEL, _CONFIG
    model, _MODEL, _CONFIG = _MODEL, None, None
    if model is not None:
        model.close()
        gc.collect()


atexit.register(_close_model)


def _ensure_model(model_path, mmproj_path, context_size, gpu_layers, attention_mode="auto"):
    global _MODEL, _CONFIG
    try:
        import llama_cpp
        from llama_cpp.llama_chat_format import Qwen35ChatHandler
    except ImportError as error:
        raise RuntimeError(
            "This node requires a vision-capable llama-cpp-python with Qwen35ChatHandler. "
            "Install it in ComfyUI's Python environment."
        ) from error

    gpu_available = llama_cpp.llama_supports_gpu_offload()
    effective_gpu_layers = gpu_layers if gpu_available else 0
    if attention_mode not in ATTENTION_MODES:
        raise ValueError(f"Unknown attention mode: {attention_mode!r}")
    config = (model_path, mmproj_path, context_size, effective_gpu_layers, attention_mode)
    if _MODEL is not None and _CONFIG == config:
        return _MODEL

    _close_model()
    if gpu_layers and not gpu_available:
        print("[Qwen GGUF] Installed llama-cpp-python has no GPU offload; using CPU.")

    handler = None
    try:
        if mmproj_path is not None:
            handler = Qwen35ChatHandler(
                mmproj_path=mmproj_path, use_gpu=gpu_available,
                enable_thinking=False, verbose=False,
            )
        model = llama_cpp.Llama(
            model_path=model_path, n_ctx=context_size,
            n_gpu_layers=effective_gpu_layers, n_batch=512,
            swa_full=True, chat_handler=handler, verbose=False,
            flash_attn_type=ATTENTION_MODES[attention_mode],
        )
    except Exception:
        if handler is not None:
            handler.close()
        raise

    _MODEL, _CONFIG = model, config
    return model


def _image_data_url(frame):
    if frame.ndim != 3 or frame.shape[-1] not in (1, 3, 4):
        raise ValueError("IMAGE must have shape [batch, height, width, 1/3/4 channels].")
    array = (frame.detach().cpu().clamp(0, 1) * 255).byte().numpy()
    image = Image.fromarray(array.squeeze(-1) if array.shape[-1] == 1 else array)
    if image.mode == "RGBA":
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image)
    image = image.convert("RGB")
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _messages(system_prompt, user_prompt, image):
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    if image is None:
        messages.append({"role": "user", "content": user_prompt})
        return messages

    if image.ndim != 4 or image.shape[0] == 0:
        raise ValueError("IMAGE must be a non-empty [batch, height, width, channels] tensor.")
    content = [{"type": "text", "text": user_prompt}]
    for index, frame in enumerate(image):
        if image.shape[0] > 1:
            content.append({"type": "text", "text": f"Image {index + 1}:"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(frame)}})
    messages.append({"role": "user", "content": content})
    return messages


def _prompt_with_preset(preset_prompt, user_prompt):
    if preset_prompt not in PROMPT_PRESETS:
        raise ValueError(f"Unknown prompt preset: {preset_prompt!r}")
    preset = PROMPT_PRESETS[preset_prompt]
    return "\n\n".join(part for part in (preset, user_prompt.strip()) if part)


class QwenGGUFInference:
    @classmethod
    def INPUT_TYPES(cls):
        files = _gguf_files()
        models = [name for name in files if "mmproj" not in Path(name).name.lower()]
        projectors = [name for name in files if "mmproj" in Path(name).name.lower()]
        return {
            "required": {
                "model": (models or ["No GGUF model found"],),
                "mmproj": (["None"] + projectors,),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "preset_prompt": (list(PROMPT_PRESETS),),
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "attention_mode": (list(ATTENTION_MODES),),
                "context_size": ("INT", {"default": 8192, "min": 512, "max": 262144}),
                "gpu_layers": ("INT", {"default": 99, "min": 0, "max": 999}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "control_after_generate": True}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "infer"
    CATEGORY = "Qwen/GGUF"

    def infer(self, model, mmproj, system_prompt, user_prompt, max_tokens, temperature,
              context_size, gpu_layers, image=None, preset_prompt="None",
              attention_mode="auto", keep_model_loaded=True, seed=0):
        model_path = _model_path(model)
        mmproj_path = None if mmproj == "None" else _model_path(mmproj)
        if image is not None and mmproj_path is None:
            raise ValueError("Image input requires the matching mmproj GGUF file.")
        messages = _messages(system_prompt, _prompt_with_preset(preset_prompt, user_prompt), image)
        with _LOCK:
            try:
                llm = _ensure_model(model_path, mmproj_path, context_size, gpu_layers, attention_mode)
                result = llm.create_chat_completion(
                    messages=messages, max_tokens=max_tokens, temperature=temperature, seed=seed,
                )
            finally:
                if not keep_model_loaded:
                    _close_model()
        try:
            message = result["choices"][0]["message"]
            answer = message.get("content") or message.get("reasoning_content")
            if isinstance(answer, list):
                answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
            if not isinstance(answer, str):
                raise ValueError("non-text response")
            return (answer,)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"Unexpected llama-cpp-python response: {str(result)[:1000]}") from error
