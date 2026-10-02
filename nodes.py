"""Minimal Qwen 3.5+ GGUF image/text inference node for ComfyUI."""

import atexit
import base64
import ctypes
import gc
import io
import json
import re
from pathlib import Path
import threading

import folder_paths
from PIL import Image


MODEL_CATEGORY = "qwen_gguf"
MODEL_DIR = Path(folder_paths.models_dir) / "LLM"
folder_paths.add_model_folder_path(MODEL_CATEGORY, str(MODEL_DIR))

_LOCK = threading.RLock()
_DOWNLOAD_LOCK = threading.RLock()
_MODEL = None
_CONFIG = None

THINKING_LEVELS = {"auto": None, "low": 256, "medium": 1024, "high": 4096, "custom": None}
_THINKING_OVERHEAD = 64
_THINKING_CONCLUSION = "\nI have reached my reasoning budget. I will now give the final answer directly and concisely.\n"

# Pinned together so the language model and vision projector come from the same
# upstream snapshot. Sizes are exact bytes reported by the Hugging Face Hub.
HF_REPO_ID = "unsloth/Qwen3.8-27B-GGUF"
HF_REVISION = "4ca720788d1e01f1bff70c033e0d0028fd02e502"
HF_MODEL_DIR = MODEL_DIR / "unsloth" / "Qwen3.8-27B-GGUF"
HF_MMPROJ = "mmproj-F16.gguf"
HF_FILE_SIZES = {
    "Qwen3.8-27B-UD-IQ2_S.gguf": 8371970048,
    "Qwen3.8-27B-UD-IQ3_XXS.gguf": 10934860704,
    "Qwen3.8-27B-UD-IQ3_S.gguf": 12040883104,
    "Qwen3.8-27B-UD-Q3_K_XL.gguf": 13146393504,
    HF_MMPROJ: 927607488,
}
DOWNLOAD_PRESETS = {
    "Download | Unsloth Qwen3.8 27B IQ2_S (8.37 GB)": "Qwen3.8-27B-UD-IQ2_S.gguf",
    "Download | Unsloth Qwen3.8 27B IQ3_XXS (10.93 GB)": "Qwen3.8-27B-UD-IQ3_XXS.gguf",
    "Download | Unsloth Qwen3.8 27B IQ3_S (12.04 GB)": "Qwen3.8-27B-UD-IQ3_S.gguf",
    "Download | Unsloth Qwen3.8 27B Q3_K_XL (13.15 GB)": "Qwen3.8-27B-UD-Q3_K_XL.gguf",
}

PROMPT_PRESETS = {
    "Empty - Nothing": "",
    "Normal - Describe": "Describe the supplied image or images accurately.",
    "Prompt Style - Tags": "Describe the visual content as concise, comma-separated tags.",
    "Prompt Style - Simple": "Write a short, clear image-generation prompt based on the visual content.",
    "Prompt Style - Detailed": "Write a detailed image-generation prompt covering subject, setting, composition, lighting, and style.",
    "Prompt Style - Extreme Detailed": "Write a very detailed image-generation prompt covering subjects, textures, composition, lighting, colors, and atmosphere.",
    "Prompt Style - Cinematic": "Write a cinematic image-generation prompt describing framing, lens perspective, lighting, mood, and scene details.",
    "Creative - Detailed Analysis": "Analyze the visual content in detail, separating direct observations from interpretation.",
    "Creative - Summarize Video": "Treat the ordered images as video frames and summarize the visible sequence of events. Do not invent unseen events.",
    "Creative - Short Story": "Write a short story inspired by the visual content; clearly treat creative details as fiction.",
    "Creative - Refine & Expand Prompt": "Refine and expand the user's prompt using details visible in the image. Preserve the user's intent.",
    "Vision - *Bounding Box": "Identify prominent objects and give approximate bounding boxes as normalized [x1,y1,x2,y2] coordinates from 0 to 1.",
    "Extract Text (OCR)": "Transcribe all readable text in the image. Preserve its original language and line breaks.",
    "Compare Images": "Compare the images in order. Explain their important similarities and differences.",
}

# llama.cpp offers Flash Attention auto/on/off, not Torch SDPA/SageAttention/FA2.
ATTENTION_MODES = {"auto": -1, "on": 1, "off": 0, "enabled": 1, "disabled": 0}
ATTENTION_CHOICES = ("auto", "on", "off")

PARAMETER_DEFAULTS = {
    "max_tokens": 1024, "top_k": 30, "top_p": 0.90, "min_p": 0.05,
    "typical_p": 1.0, "temperature": 0.80, "repeat_penalty": 1.0,
    "frequency_penalty": 0.0, "presence_penalty": 0.0,
    "mirostat_mode": 0, "mirostat_eta": 0.10, "mirostat_tau": 5.0,
    "enable_thinking": False, "reasoning_budget": -1,
    "inference_mode": "auto", "max_frames": 24, "max_size": 256,
    "video_fps": 0.0,
    "image_max_tokens": -1,
    "n_batch": 512, "n_threads": 0,
    "mtp_draft_tokens": 0, "mtp_draft_p_min": 0.0,
}


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


def _valid_remote_gguf(path, expected_size):
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    with path.open("rb") as file:
        return file.read(4) == b"GGUF"


def _download_remote_file(filename):
    """Reuse a complete local file or let huggingface_hub resume its download."""
    expected_size = HF_FILE_SIZES[filename]
    target = HF_MODEL_DIR / filename
    if _valid_remote_gguf(target, expected_size):
        return str(target.resolve())
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("Automatic model downloads require huggingface-hub. Install requirements.txt.") from error

    HF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Qwen GGUF] Downloading {HF_REPO_ID}/{filename} to {HF_MODEL_DIR}")
    try:
        downloaded = Path(hf_hub_download(
            repo_id=HF_REPO_ID, filename=filename, revision=HF_REVISION,
            local_dir=str(HF_MODEL_DIR),
        ))
        if not _valid_remote_gguf(downloaded, expected_size):
            # A stale local-dir cache can report a damaged file as complete.
            downloaded = Path(hf_hub_download(
                repo_id=HF_REPO_ID, filename=filename, revision=HF_REVISION,
                local_dir=str(HF_MODEL_DIR), force_download=True,
            ))
        if not _valid_remote_gguf(downloaded, expected_size):
            raise RuntimeError(f"Downloaded file is incomplete or invalid: {downloaded}")
    except Exception as error:
        raise RuntimeError(
            f"Could not download {filename} from https://huggingface.co/{HF_REPO_ID}. "
            f"Check network access and free disk space, then retry; interrupted downloads can resume."
        ) from error
    return str(downloaded.resolve())


def _download_preset(choice):
    with _DOWNLOAD_LOCK:
        model_path = _download_remote_file(DOWNLOAD_PRESETS[choice])
        mmproj_path = _download_remote_file(HF_MMPROJ)
    return model_path, mmproj_path


def _close_model():
    global _MODEL, _CONFIG
    model, _MODEL, _CONFIG = _MODEL, None, None
    if model is not None:
        model.close()
        gc.collect()


atexit.register(_close_model)


def _gpu_offload_available(llama_cpp):
    """Register dynamic ggml backends before asking whether CUDA is available."""
    if llama_cpp.llama_supports_gpu_offload():
        return True
    try:
        from llama_cpp._ggml import ggml_backend_load_all_from_path
    except ImportError:
        return False
    # On Windows, importing torch makes its bundled CUDA runtime DLLs available
    # to ggml-cuda.dll. ComfyUI normally imports torch before executing nodes.
    import torch

    llama_cpp.llama_backend_init()
    lib_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
    if lib_dir.is_dir():
        ggml_backend_load_all_from_path(ctypes.c_char_p(str(lib_dir).encode("utf-8")))
    return llama_cpp.llama_supports_gpu_offload()


def _ensure_model(model_path, mmproj_path, context_size, gpu_layers, attention_mode="auto",
                  *, n_batch=512, n_ubatch=512, n_threads=0, image_max_tokens=-1,
                  enable_thinking=False, mtp_draft_tokens=0, mtp_draft_p_min=0.0):
    global _MODEL, _CONFIG
    try:
        import llama_cpp
    except ImportError as error:
        raise RuntimeError(
            "This node requires a vision-capable llama-cpp-python. Install it in ComfyUI's Python environment."
        ) from error

    use_generic_handler = "qwen3.8" in Path(model_path).name.lower()
    gpu_available = _gpu_offload_available(llama_cpp) if gpu_layers else False
    effective_gpu_layers = gpu_layers if gpu_available else 0
    if attention_mode not in ATTENTION_MODES:
        raise ValueError(f"Unknown attention mode: {attention_mode!r}")
    if n_ubatch > n_batch:
        raise ValueError("n_ubatch cannot exceed n_batch.")
    config = (model_path, mmproj_path, context_size, effective_gpu_layers, attention_mode,
              n_batch, n_ubatch, n_threads, image_max_tokens, enable_thinking,
              mtp_draft_tokens, mtp_draft_p_min, use_generic_handler)
    if _MODEL is not None and _CONFIG == config:
        return _MODEL

    _close_model()
    if gpu_layers and not gpu_available:
        print("[Qwen GGUF] Installed llama-cpp-python has no GPU offload; using CPU.")

    handler = None
    try:
        if mmproj_path is not None and not use_generic_handler:
            try:
                from llama_cpp.llama_chat_format import Qwen35ChatHandler
            except ImportError as error:
                raise RuntimeError("This model requires a llama-cpp-python build with Qwen35ChatHandler.") from error
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
        model_options = dict(
            model_path=model_path, n_ctx=context_size,
            n_gpu_layers=effective_gpu_layers, n_batch=n_batch, n_ubatch=n_ubatch,
            n_threads=n_threads or None, speculative=speculative,
            swa_full=True, chat_handler=handler, verbose=False,
            flash_attn_type=ATTENTION_MODES[attention_mode],
        )
        if use_generic_handler and mmproj_path is not None:
            model_options["mmproj_path"] = mmproj_path
            model_options["chat_handler_kwargs"] = {
                "use_gpu": gpu_available, "image_max_tokens": image_max_tokens,
                "extra_template_arguments": {"enable_thinking": enable_thinking},
                "verbose": False,
            }
        model = llama_cpp.Llama(**model_options)
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


def _image_data_url(frame, max_size=0):
    if frame.ndim != 3 or frame.shape[-1] not in (1, 3, 4):
        raise ValueError("IMAGE must have shape [batch, height, width, 1/3/4 channels].")
    array = (frame.detach().cpu().clamp(0, 1) * 255).byte().numpy()
    image = Image.fromarray(array.squeeze(-1) if array.shape[-1] == 1 else array)
    if image.mode == "RGBA":
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image)
    image = image.convert("RGB")
    if max_size and max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _select_frames(image, max_frames):
    if image.ndim != 4 or image.shape[0] == 0:
        raise ValueError("IMAGE input must be a non-empty [batch, height, width, channels] tensor.")
    total = image.shape[0]
    count = min(total, max_frames)
    indices = ([0] if count == 1 else
               [round(position * (total - 1) / (count - 1)) for position in range(count)])
    selected = set(indices)
    return [(index, frame) for index, frame in enumerate(image) if index in selected]


def _messages(system_prompt, user_prompt, frames=None, max_size=0, frame_rate=None,
              as_video=False):
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    if frames is None:
        messages.append({"role": "user", "content": user_prompt})
        return messages

    content = [{"type": "text", "text": user_prompt}]
    for index, frame in frames:
        if frame_rate is not None:
            content.append({"type": "text", "text": f"Frame {index + 1} ({index / frame_rate:.2f}s):"})
        elif as_video:
            content.append({"type": "text", "text": f"Frame {index + 1}:"})
        elif len(frames) > 1:
            content.append({"type": "text", "text": f"Image {index + 1}:"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(frame, max_size)}})
    messages.append({"role": "user", "content": content})
    return messages


def _response_parts(result):
    try:
        message = result["choices"][0]["message"]
        answer = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if isinstance(answer, list):
            answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
        if not isinstance(answer, str) or not isinstance(reasoning, str):
            raise ValueError("non-text response")
        # Qwen templates may put the opening tag in the prompt, so only the
        # closing tag appears in generated content. Also hide unfinished blocks.
        parts = re.split(r"(<\s*/?\s*think\s*>)", answer, flags=re.IGNORECASE)
        if len(parts) > 1:
            in_thought = bool(re.fullmatch(r"<\s*/\s*think\s*>", parts[1], re.IGNORECASE))
            final_parts, thought_parts = [], []
            for part in parts:
                if re.fullmatch(r"<\s*think\s*>", part, re.IGNORECASE):
                    in_thought = True
                elif re.fullmatch(r"<\s*/\s*think\s*>", part, re.IGNORECASE):
                    in_thought = False
                else:
                    (thought_parts if in_thought else final_parts).append(part)
            reasoning = reasoning or "\n\n".join(p.strip() for p in thought_parts if p.strip())
            answer = "".join(final_parts).strip()
        if not answer.strip() and reasoning and result["choices"][0].get("finish_reason") == "length":
            raise RuntimeError(
                "The model reached max_tokens during thinking and produced no final answer. "
                "Increase max_tokens or disable enable_thinking on the inference node."
            )
        return answer, reasoning
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(f"Unexpected llama-cpp-python response: {str(result)[:1000]}") from error


def _mode(selected_count, params, context_size, has_video=False):
    requested = params["inference_mode"].replace(" ", "_")
    if requested not in ("auto", "images", "video", "one_by_one"):
        raise ValueError(f"Unknown inference_mode: {requested!r}")
    if requested != "auto":
        return requested
    estimated_image_tokens = params["image_max_tokens"] if params["image_max_tokens"] > 0 else 512
    if selected_count <= 8 and selected_count * estimated_image_tokens <= context_size * 0.6:
        return "video" if has_video else "images"
    return "one_by_one"


def _prompt_with_preset(preset_prompt, user_prompt):
    if preset_prompt not in PROMPT_PRESETS:
        raise ValueError(f"Unknown prompt preset: {preset_prompt!r}")
    preset = PROMPT_PRESETS[preset_prompt]
    return "\n\n".join(part for part in (preset, user_prompt.strip()) if part)


def _thinking_budget(level, params):
    if level not in THINKING_LEVELS:
        raise ValueError(f"Unknown thinking level: {level!r}")
    if not params["enable_thinking"]:
        return 0
    if level == "custom":
        budget = params["reasoning_budget"]
        if not isinstance(budget, int) or budget < -1:
            raise ValueError("reasoning_budget must be -1, 0, or a positive integer.")
        return budget
    if level == "auto":
        return min(1024, max(128, params["max_tokens"] // 2))
    return THINKING_LEVELS[level]


def _complete(llm, messages, options, answer_tokens, budget, thinking, call_stats, stage):
    """Apply the budget after MTMD has counted the actual text/media prompt.

    The caller holds _LOCK. The temporary adapter is restored even on errors.
    This keeps final-answer space available when the backend clamps generation
    to the context window, and detects templates that already opened <think>.
    """
    original = llm.create_completion
    had_override = "create_completion" in vars(llm)

    def bounded_completion(*args, **kwargs):
        prompt = kwargs.get("prompt", args[0] if args else None)
        if isinstance(prompt, str):
            tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        else:
            tokens = list(prompt)
        available = llm.n_ctx() - len(tokens)
        if available <= 0:
            raise ValueError("Input fills the context window. Increase context_size or reduce max_frames/max_size.")
        effective_budget = budget
        if thinking and budget >= 0:
            if available <= _THINKING_OVERHEAD:
                raise ValueError("No room for thinking and a final answer. Increase context_size or reduce the input.")
            effective_budget = min(budget, max(0, available - answer_tokens - _THINKING_OVERHEAD))
            limit = min(available, answer_tokens + effective_budget + _THINKING_OVERHEAD)
        else:
            # Custom -1 keeps the legacy unrestricted budget within max_tokens.
            limit = min(available, answer_tokens)
        # Only text tokens are detokenized; MTMD uses negative IDs for media.
        tail = llm.detokenize([t for t in tokens[-128:] if t >= 0], special=True).decode("utf-8", errors="replace")
        start_in_prompt = tail.rfind("<think>") > tail.rfind("</think>")
        kwargs.update(max_tokens=limit, reasoning_budget=effective_budget,
                      reasoning_start_in_prompt=start_in_prompt,
                      reasoning_start="<think>", reasoning_end="</think>",
                      reasoning_budget_message=_THINKING_CONCLUSION if thinking and effective_budget >= 0 else None)
        call_stats.append({"stage": stage, "prompt_tokens": len(tokens),
                           "reasoning_budget": effective_budget, "max_tokens": limit,
                           "reasoning_start_in_prompt": start_in_prompt})
        return original(*args, **kwargs)

    llm.create_completion = bounded_completion
    try:
        # Requests in this node are stateless. The installed Qwen hybrid backend
        # can reuse a prompt checkpoint with a stale last-decode cursor; reset
        # context for every independent call, retaining the loaded weights.
        llm.reset()
        planned_limit = answer_tokens + budget + _THINKING_OVERHEAD if thinking and budget >= 0 else answer_tokens
        result = llm.create_chat_completion(messages=messages, max_tokens=planned_limit,
                                            reasoning_budget=budget, **options)
        choice = result["choices"][0]
        content = choice["message"].get("content")
        if (thinking and call_stats[-1]["reasoning_start_in_prompt"] and
                choice.get("finish_reason") == "length" and isinstance(content, str) and
                not re.search(r"<\s*/?\s*think\s*>", content, re.IGNORECASE) and
                not choice["message"].get("reasoning_content") and not choice["message"].get("reasoning")):
            # The opening tag was in the prompt and generation never reached its
            # closing tag. Mark it for the parser so truncated thoughts cannot leak.
            result = dict(result, choices=[dict(choice, message=dict(choice["message"], content="<think>" + content))])
        return result
    finally:
        if had_override:
            llm.create_completion = original
        else:
            del llm.create_completion


class QwenGGUFParameters:
    """A reusable parameter bundle for the inference node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enable_thinking": ("BOOLEAN", {"default": False}),
            "reasoning_budget": ("INT", {"default": -1, "min": -1, "max": 32768,
                "tooltip": "Used when the main thinking_level is custom. 0 disables thinking; -1 removes the thinking budget within the total max_tokens limit."}),
            "inference_mode": (["auto", "one by one", "images", "video"],),
            "max_frames": ("INT", {"default": 24, "min": 1, "max": 1024}),
            "max_size": ("INT", {"default": 256, "min": 0, "max": 4096}),
            "video_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.1}),
            "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768,
                "tooltip": "Without thinking: total output limit. With a finite thinking budget: final-answer space to reserve; thinking tokens are added to the total limit, subject to context_size."}),
            "top_k": ("INT", {"default": 30, "min": 0, "max": 1000}),
            "top_p": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
            "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "temperature": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 2.0, "step": 0.05}),
            "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
            "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
            "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
            "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 2}),
            "mirostat_eta": ("FLOAT", {"default": 0.10, "min": 0.001, "max": 1.0, "step": 0.01}),
            "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            "image_max_tokens": ("INT", {"default": -1, "min": -1, "max": 16384}),
            "n_batch": ("INT", {"default": 512, "min": 32, "max": 8192}),
            "n_threads": ("INT", {"default": 0, "min": 0, "max": 128}),
            "mtp_draft_tokens": ("INT", {"default": 0, "min": 0, "max": 32}),
            "mtp_draft_p_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("QWEN_GGUF_PARAMETERS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "build"
    CATEGORY = "Qwen/GGUF"

    def build(self, **options):
        return (dict(options),)


class QwenGGUFInference:
    @classmethod
    def INPUT_TYPES(cls):
        files = _gguf_files()
        models = [name for name in files if "mmproj" not in Path(name).name.lower()]
        projectors = [name for name in files if "mmproj" in Path(name).name.lower()]
        model_choices = (models or ["Select a local model or download preset"]) + list(DOWNLOAD_PRESETS)
        return {
            "required": {
                "model": (model_choices,),
                "mmproj": (["None"] + projectors,),
                "preset_prompt": (list(PROMPT_PRESETS), {"default": "Normal - Describe"}),
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "attention_mode": (list(ATTENTION_CHOICES),),
                "context_size": ("INT", {"default": 8192, "min": 512, "max": 262144}),
                "gpu_layers": ("INT", {"default": 99, "min": 0, "max": 999}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "control_after_generate": True}),
                "enable_thinking": ("BOOLEAN", {"default": False,
                    "tooltip": "Enable model thinking. This main-node switch overrides the Parameters setting."}),
                "thinking_level": (list(THINKING_LEVELS), {"default": "auto",
                    "tooltip": "Thinking token budget: low 256, medium 1024, high 4096; auto adapts to max_tokens; custom uses Parameters reasoning_budget. Requires enable_thinking."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "parameters": ("QWEN_GGUF_PARAMETERS",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "reasoning", "stats_json")
    FUNCTION = "infer"
    CATEGORY = "Qwen/GGUF"

    def infer(self, model, mmproj, user_prompt, system_prompt, preset_prompt="Normal - Describe",
              attention_mode="auto", context_size=8192, gpu_layers=99,
              keep_model_loaded=True, seed=0, image=None, video=None, parameters=None,
              enable_thinking=None, thinking_level=None):
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError("parameters must come from a Qwen GGUF Parameters node.")
        params = dict(PARAMETER_DEFAULTS)
        if parameters:
            unknown = set(parameters) - set(params)
            if unknown:
                raise ValueError(f"Unknown inference parameters: {sorted(unknown)}")
            params.update(parameters)
        # Keep the old Parameters field compatible; the visible main-node
        # switch takes precedence whenever supplied by ComfyUI.
        if enable_thinking is not None:
            params["enable_thinking"] = bool(enable_thinking)
        # Calls from older workflows retain their explicit Parameters budget.
        level = thinking_level if thinking_level is not None else "custom"
        budget = _thinking_budget(level, params)
        if budget == 0:
            params["enable_thinking"] = False
        if image is not None and video is not None:
            raise ValueError("Connect either image or video input, not both.")
        if params["max_frames"] < 1 or params["max_size"] < 0 or params["video_fps"] < 0:
            raise ValueError("max_frames must be positive; max_size and video_fps cannot be negative.")
        media = image
        frame_rate = float(params["video_fps"]) if image is not None and params["video_fps"] else None
        if video is not None:
            if not callable(getattr(video, "get_components", None)):
                raise TypeError("video input must be a ComfyUI VIDEO with get_components().")
            components = video.get_components()
            media = components.images
            frame_rate = float(components.frame_rate)
            if frame_rate <= 0:
                raise ValueError("VIDEO frame rate must be positive.")
        selected = _select_frames(media, params["max_frames"]) if media is not None else None
        selected_count = len(selected) if selected is not None else 0
        is_video_sequence = video is not None or frame_rate is not None or params["inference_mode"] == "video"
        mode = _mode(selected_count, params, context_size, is_video_sequence)
        if model in DOWNLOAD_PRESETS:
            model_path, mmproj_path = _download_preset(model)
        else:
            model_path = _model_path(model)
            mmproj_path = None if mmproj == "None" else _model_path(mmproj)
        if media is not None and mmproj_path is None:
            raise ValueError("IMAGE input requires the matching mmproj GGUF file.")
        if params["enable_thinking"] and mmproj_path is None:
            raise ValueError("enable_thinking requires mmproj: the Qwen35ChatHandler controls this template option.")
        if params["mtp_draft_tokens"] >= params["n_batch"]:
            raise ValueError("mtp_draft_tokens must be smaller than n_batch.")

        prompt = _prompt_with_preset(preset_prompt, user_prompt)
        completion_options = dict(
            temperature=params["temperature"], top_p=params["top_p"],
            top_k=params["top_k"], min_p=params["min_p"],
            typical_p=params["typical_p"], repeat_penalty=params["repeat_penalty"],
            frequency_penalty=params["frequency_penalty"],
            presence_penalty=params["presence_penalty"],
            mirostat_mode=params["mirostat_mode"],
            mirostat_tau=params["mirostat_tau"], mirostat_eta=params["mirostat_eta"],
            seed=seed,
        )
        results = []
        thinking_calls = []
        with _LOCK:
            try:
                llm = _ensure_model(
                    model_path, mmproj_path, context_size, gpu_layers, attention_mode,
                    n_batch=params["n_batch"], n_ubatch=min(512, params["n_batch"]),
                    n_threads=params["n_threads"], image_max_tokens=params["image_max_tokens"],
                    enable_thinking=params["enable_thinking"],
                    mtp_draft_tokens=params["mtp_draft_tokens"],
                    mtp_draft_p_min=params["mtp_draft_p_min"],
                )
                effective_gpu_layers = _CONFIG[3]
                if mode == "one_by_one" and selected_count > 1:
                    observations = []
                    for index, frame in selected:
                        label = (f"Frame {index + 1} ({index / frame_rate:.2f}s)"
                                 if frame_rate is not None else
                                 f"Frame {index + 1}" if is_video_sequence else f"Image {index + 1}")
                        frame_prompt = (f"{prompt}\n\nDescribe only {label}. Keep the visible observations "
                                        "concise; do not infer events outside this frame.")
                        frame_messages = _messages(system_prompt, frame_prompt, [(index, frame)],
                                                   params["max_size"], frame_rate,
                                                   as_video=is_video_sequence)
                        frame_budget = min(budget, 64) if budget >= 0 else 64
                        frame_result = _complete(
                            llm, frame_messages, completion_options, min(params["max_tokens"], 256),
                            frame_budget if params["enable_thinking"] else 0,
                            params["enable_thinking"], thinking_calls, "frame",
                        )
                        results.append(frame_result)
                        observation, _ = _response_parts(frame_result)
                        observations.append(f"{label}: {observation[:800]}")
                    summary_prompt = (f"{prompt}\n\nThe following are observations from ordered "
                                      "frames/images. Answer the original request using them. "
                                      "Distinguish visible changes from uncertain motion or unseen events.\n\n" +
                                      "\n".join(observations))
                    messages = _messages(system_prompt, summary_prompt)
                else:
                    request_system_prompt = system_prompt
                    if mode == "video" and selected_count:
                        request_system_prompt = "\n\n".join(part for part in (
                            system_prompt.strip(),
                            "Treat the ordered frames as one video sequence. Describe only visible "
                            "changes; do not invent motion or events between sampled frames.",
                        ) if part)
                    messages = _messages(request_system_prompt, prompt, selected,
                                         params["max_size"], frame_rate if mode == "video" else None,
                                         as_video=mode == "video")
                result = _complete(
                    llm, messages, completion_options, params["max_tokens"], budget,
                    params["enable_thinking"], thinking_calls, "final",
                )
                results.append(result)
                spec_stats = getattr(llm, "last_speculative_stats", {}) if params["mtp_draft_tokens"] else {}
            finally:
                if not keep_model_loaded:
                    _close_model()

        answer, reasoning = _response_parts(result)
        usage = {}
        for item in results:
            for key, value in item.get("usage", {}).items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
        stats = {"usage": usage, "mtp": spec_stats, "inference_mode": mode,
                 "thinking": {"enabled": params["enable_thinking"], "level": level,
                              "budget_requested": budget, "calls": thinking_calls},
                 "gpu_layers_effective": effective_gpu_layers,
                 "total_frames": int(media.shape[0]) if media is not None else 0,
                 "selected_frames": selected_count, "model_calls": len(results)}
        return (answer, reasoning, json.dumps(stats, ensure_ascii=False, default=str))
