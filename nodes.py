"""Minimal Qwen 3.5+ GGUF image/text inference node for ComfyUI."""

import atexit
import base64
import gc
import io
import json
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


def _ensure_model(model_path, mmproj_path, context_size, gpu_layers, attention_mode="auto",
                  *, n_batch=512, n_ubatch=512, n_threads=0, image_max_tokens=-1,
                  enable_thinking=False, mtp_draft_tokens=0, mtp_draft_p_min=0.0):
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
    if n_ubatch > n_batch:
        raise ValueError("n_ubatch cannot exceed n_batch.")
    config = (model_path, mmproj_path, context_size, effective_gpu_layers, attention_mode,
              n_batch, n_ubatch, n_threads, image_max_tokens, enable_thinking,
              mtp_draft_tokens, mtp_draft_p_min)
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
                enable_thinking=enable_thinking, image_max_tokens=image_max_tokens,
                verbose=False,
            )
        speculative = None
        if mtp_draft_tokens:
            try:
                from llama_cpp.llama_speculative import SpecConfig, SpeculativeType
            except ImportError as error:
                raise RuntimeError("This llama-cpp-python build does not provide MTP speculative decoding.") from error
            speculative = SpecConfig(
                spec_type=SpeculativeType.DRAFT_MTP,
                draft_n_max=mtp_draft_tokens,
                draft_p_min=mtp_draft_p_min,
            )
        model = llama_cpp.Llama(
            model_path=model_path, n_ctx=context_size,
            n_gpu_layers=effective_gpu_layers, n_batch=n_batch, n_ubatch=n_ubatch,
            n_threads=n_threads or None, speculative=speculative,
            swa_full=True, chat_handler=handler, verbose=False,
            flash_attn_type=ATTENTION_MODES[attention_mode],
        )
    except Exception as error:
        if handler is not None:
            handler.close()
        if mtp_draft_tokens:
            raise RuntimeError(
                "MTP initialization failed. Check that the selected GGUF contains compatible "
                "MTP layers and that this llama-cpp-python build supports draft-mtp."
            ) from error
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


def _messages(system_prompt, user_prompt, image, max_images=0):
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    if image is None:
        messages.append({"role": "user", "content": user_prompt})
        return messages

    if image.ndim != 4 or image.shape[0] == 0:
        raise ValueError("IMAGE must be a non-empty [batch, height, width, channels] tensor.")
    total = image.shape[0]
    count = min(total, max_images) if max_images else total
    indices = ([0] if count == 1 else
               [round(position * (total - 1) / (count - 1)) for position in range(count)])
    selected = set(indices)
    content = [{"type": "text", "text": user_prompt}]
    for index, frame in enumerate(image):
        if index not in selected:
            continue
        if count > 1:
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


class QwenGGUFInferenceAdvanced(QwenGGUFInference):
    """Expose GGUF load options, samplers, thinking, and built-in MTP."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 40, "min": 0, "max": 1000}),
            "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
            "penalty_last_n": ("INT", {"default": 64, "min": -1, "max": 8192}),
            "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
            "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
            "mirostat_mode": (["off", "v1", "v2"],),
            "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
            "enable_thinking": ("BOOLEAN", {"default": False}),
            "reasoning_budget": ("INT", {"default": -1, "min": -1, "max": 32768}),
            "max_images": ("INT", {"default": 0, "min": 0, "max": 1024}),
            "image_max_tokens": ("INT", {"default": -1, "min": -1, "max": 16384}),
            "n_batch": ("INT", {"default": 512, "min": 32, "max": 8192}),
            "n_ubatch": ("INT", {"default": 512, "min": 32, "max": 8192}),
            "n_threads": ("INT", {"default": 0, "min": 0, "max": 128}),
            "mtp_draft_tokens": ("INT", {"default": 0, "min": 0, "max": 32}),
            "mtp_draft_p_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return inputs

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "reasoning", "stats_json")
    FUNCTION = "infer"

    def infer(self, **options):
        model_path = _model_path(options["model"])
        mmproj_path = None if options["mmproj"] == "None" else _model_path(options["mmproj"])
        image = options.get("image")
        if image is not None and mmproj_path is None:
            raise ValueError("Image input requires the matching mmproj GGUF file.")
        if options["enable_thinking"] and mmproj_path is None:
            raise ValueError("enable_thinking requires mmproj: the Qwen35ChatHandler controls this template option.")
        if options["mtp_draft_tokens"] and options["mtp_draft_tokens"] >= options["n_batch"]:
            raise ValueError("mtp_draft_tokens must be smaller than n_batch.")

        messages = _messages(
            options["system_prompt"],
            _prompt_with_preset(options["preset_prompt"], options["user_prompt"]),
            image, options["max_images"],
        )
        with _LOCK:
            try:
                llm = _ensure_model(
                    model_path, mmproj_path, options["context_size"], options["gpu_layers"],
                    options["attention_mode"], n_batch=options["n_batch"],
                    n_ubatch=options["n_ubatch"], n_threads=options["n_threads"],
                    image_max_tokens=options["image_max_tokens"],
                    enable_thinking=options["enable_thinking"],
                    mtp_draft_tokens=options["mtp_draft_tokens"],
                    mtp_draft_p_min=options["mtp_draft_p_min"],
                )
                result = llm.create_chat_completion(
                    messages=messages, max_tokens=options["max_tokens"],
                    temperature=options["temperature"], top_p=options["top_p"],
                    top_k=options["top_k"], min_p=options["min_p"],
                    repeat_penalty=options["repetition_penalty"],
                    penalty_last_n=options["penalty_last_n"],
                    frequency_penalty=options["frequency_penalty"],
                    presence_penalty=options["presence_penalty"],
                    mirostat_mode={"off": 0, "v1": 1, "v2": 2}[options["mirostat_mode"]],
                    mirostat_tau=options["mirostat_tau"], mirostat_eta=options["mirostat_eta"],
                    reasoning_budget=options["reasoning_budget"] if options["enable_thinking"] else -1,
                    seed=options["seed"],
                )
                spec_stats = getattr(llm, "last_speculative_stats", {}) if options["mtp_draft_tokens"] else {}
            finally:
                if not options["keep_model_loaded"]:
                    _close_model()

        try:
            message = result["choices"][0]["message"]
            answer = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            if isinstance(answer, list):
                answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
            if not isinstance(answer, str) or not isinstance(reasoning, str):
                raise ValueError("non-text response")
            if "<think>" in answer and "</think>" in answer:
                prefix, rest = answer.split("<think>", 1)
                thought, suffix = rest.split("</think>", 1)
                reasoning = reasoning or thought.strip()
                answer = (prefix + suffix).strip()
            stats = {"usage": result.get("usage", {}), "mtp": spec_stats}
            return (answer, reasoning, json.dumps(stats, ensure_ascii=False, default=str))
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"Unexpected llama-cpp-python response: {str(result)[:1000]}") from error
