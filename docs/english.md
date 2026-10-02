# English quick start

[Home](../README.md) · [Installation guide (Chinese)](installation.md) · [Usage guide (Chinese)](usage.md) · [Troubleshooting (Chinese)](troubleshooting.md)

Run Qwen 3.5+ GGUF models inside ComfyUI for text, images, image batches, and video-frame analysis. The extension provides **Qwen 3.5+ GGUF Inference** and an optional **Qwen GGUF Parameters** node. No separate llama-server is needed.

## Installation

Use the Python interpreter that runs ComfyUI. For Windows portable, set your real paths in PowerShell:

```powershell
$ComfyRoot = 'C:\AI\ComfyUI_windows_portable\ComfyUI'
$ComfyPython = 'C:\AI\ComfyUI_windows_portable\python_embeded\python.exe'
git clone https://github.com/Bandukids/ComfyUI-Qwen-gguf.git "$ComfyRoot\custom_nodes\ComfyUI-Qwen-gguf"
& $ComfyPython -m pip install -r "$ComfyRoot\custom_nodes\ComfyUI-Qwen-gguf\requirements.txt"
```

Install a vision-capable wheel from [JamePeng/llama-cpp-python Releases](https://github.com/JamePeng/llama-cpp-python/releases), matching your OS, architecture, Python version, and GPU backend. Replace the placeholder with the actual downloaded wheel filename:

```powershell
& $ComfyPython -m pip install 'C:\Downloads\<your-wheel-filename>.whl'
& $ComfyPython -c "import llama_cpp; from llama_cpp.llama_chat_format import Qwen35ChatHandler; from llama_cpp.llama_multimodal import GenericMTMDChatHandler; print(llama_cpp.__version__); print('Vision handlers OK')"
& $ComfyPython -c "import inspect; from llama_cpp import Llama; print('Reasoning budget API:', 'reasoning_budget' in inspect.signature(Llama.create_completion).parameters)"
```

The requirements file installs Pillow and Hugging Face Hub. The GGUF backend is installed separately to avoid replacing an existing GPU build. Restart ComfyUI after installation.

For Linux/macOS or a virtual environment, run the equivalent commands using ComfyUI's active Python and a platform-compatible backend; see [upstream installation](https://github.com/JamePeng/llama-cpp-python#installation).

## Models and matching mmproj

Place GGUF files anywhere under `ComfyUI/models/LLM/`. Projector filenames must contain `mmproj` to appear in that dropdown. Preserve the model-family identifier in the model filename.

**Use the mmproj specified by the model publisher. Qwen3.5 9B must not be paired with a Qwen3.8 27B projector.** Main-model and projector quantization labels do not have to match: a Q8 main model may use its matching F16/BF16 projector.

Alternatively, select one of the four `Download | Unsloth Qwen3.8 27B ...` presets. Execution downloads only the selected quantization plus its matching projector from the pinned [Unsloth repository](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main). Completed files are reused and interrupted downloads can resume. The sizes shown are download sizes, not VRAM requirements.

## First image

1. Add **Qwen 3.5+ GGUF Inference** from **Qwen/GGUF**.
2. Select a main model and matching mmproj, or a download preset.
3. Connect a loaded image's `IMAGE` output to `image`.
4. Enter a user prompt; leave thinking disabled for the first run.
5. Connect `response` to any `STRING` text-display node from your installed extensions, then execute.

The extension does not include its own text-display node. `response` contains the final answer; `reasoning` contains separate thinking text; `stats_json` contains debugging statistics. Text-only input requires no media connection. Text-only, non-thinking inference can use `mmproj=None`; thinking in this node requires a matching mmproj.

## Batches and video

Connect either `image: IMAGE` or native `video: VIDEO`, not both. IMAGE accepts still images, batches, and decoded video frames. Set `inference_mode=video` for video-frame batches, or supply their actual `video_fps` and use `auto`. Native VIDEO reads its own frame rate.

`max_frames=24` samples across the entire input sequence when exceeded, including still-image batches. Default `max_size=256` limits the longest edge. Audio is not analyzed. Native VIDEO may decode all frames before sampling, so split long clips upstream.

`images/video` use one request for selected frames. `one by one` observes each frame and then summarizes, normally N+1 requests. `auto` often chooses the latter above eight selected frames; inspect `model_calls` if execution is slow.

## Thinking controls

Enable `enable_thinking` on the main node, then choose:

| Level | Thinking token budget |
| --- | --- |
| `auto` | Half of max_tokens, bounded to 128–1024 |
| `low` | 256 |
| `medium` | 1024 |
| `high` | 4096 |
| `custom` | Parameters reasoning_budget; 0 disables, -1 leaves it unrestricted within max_tokens |

With a finite budget, generation reserves answer space and adds thinking plus 64 tokens of closing overhead. Actual context space may reduce the thinking budget. Individual frame observations are capped at 64 thinking tokens; the final summary uses the selected level. Higher budgets cost more time and do not guarantee better answers.

Changing the level reuses loaded model weights. Use `keep_model_loaded=false` when you need to release the GGUF before large downstream image-generation models.

## Checks and troubleshooting

Local model checks cover Qwen3.5 9B on Windows/CUDA with vision backend 0.3.49. Unit tests cover other control paths; all models, platforms, and MTP combinations have not been tested.

- Missing nodes: verify the installation folder, Python environment, and startup import log.
- `IMAGE input requires ... mmproj`: select the matching projector.
- `Failed to load mtmd context`: check model/projector pairing, file completeness, backend support, and nearby memory-allocation logs.
- High CPU load: verify the llama GPU backend and layer-offload logs; check frame count and thinking budget.
- Updating: pull the repository, reinstall requirements, restart ComfyUI, and refresh the browser. Recreate an old node if new widgets are absent.

Report issues with model/projector filenames, backend wheel/version, GPU, settings, and the relevant console log through [GitHub Issues](https://github.com/Bandukids/ComfyUI-Qwen-gguf/issues).
