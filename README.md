# ComfyUI Qwen GGUF

**在 ComfyUI 中用本地 Qwen 3.5+ GGUF 进行文本、图片批次与视频帧推理。** 只保留一个推理节点和一个可选的参数节点，无需 `llama-server`、Transformers 或在线 API。

> Local Qwen GGUF vision inference for ComfyUI. Accepts images, ordered image batches, and native ComfyUI video. See the installation steps below.

## 功能一览

| 功能 | 说明 |
| --- | --- |
| 模型 | 本地 GGUF 或四个可自动下载的 Unsloth Qwen3.8 27B 预设；视觉输入需配套 mmproj |
| 输入 | 文本、单图、多图、图片形式的视频帧批次、原生 ComfyUI `VIDEO` |
| 提示词 | 用户提示词、系统提示词，以及描述、OCR、视频总结等预设 |
| 推理 | `auto`、`images`、`video`、`one by one` 四种模式 |
| 调参 | 采样、思考预算、上下文、Flash Attention、GPU 层数、可选 MTP |
| 输出 | `response`、`reasoning`、`stats_json` |

Qwen3.5 使用 `Qwen35ChatHandler`，Qwen3.8 使用模型聊天模板驱动的 `GenericMTMDChatHandler`。其他后续模型是否兼容，取决于 GGUF、匹配的 mmproj 和所安装的 `llama-cpp-python` 构建；不支持早期 Qwen 模型。

## 安装

### 1. 放入自定义节点目录

```text
ComfyUI/
├─ custom_nodes/
│  └─ ComfyUI-Qwen-gguf/
└─ models/
   └─ LLM/
      └─ Qwen-VL/
         ├─ your-model.gguf
         └─ mmproj-your-model.gguf
```

主模型和 **与它匹配** 的视觉投影文件可以放在 `ComfyUI/models/LLM/` 的任意子目录。纯文本推理可将 `mmproj` 设为 `None`。

### 2. 安装通用依赖

在 **ComfyUI 目录**中，使用 ComfyUI 自己的 Python 运行：

```powershell
python -m pip install -r "custom_nodes/ComfyUI-Qwen-gguf/requirements.txt"
```

Windows 便携版请把上面的 `python` 换成其自带的 `python.exe` 完整路径。`requirements.txt` 安装 Pillow 和 Hugging Face Hub 下载库；PyTorch 由 ComfyUI 提供。

### 3. 单独安装视觉版 `llama-cpp-python`

节点需要包含 `Qwen35ChatHandler` 与 `GenericMTMDChatHandler` 的构建。请从 [JamePeng/llama-cpp-python Releases](https://github.com/JamePeng/llama-cpp-python/releases) 下载与 **操作系统、Python 版本和 GPU 后端** 匹配的 wheel，再用同一个 ComfyUI Python 安装：

```powershell
python -m pip install "<下载的 wheel 文件路径>"
python -c "from llama_cpp.llama_chat_format import Qwen35ChatHandler; from llama_cpp.llama_multimodal import GenericMTMDChatHandler; print('Vision handlers OK')"
```

已用该项目的 `0.3.49` 构建验证 Qwen 3.5 视觉推理和 CUDA 后端。已有可用的 CUDA wheel 时，无需重新安装；也不要仅为满足依赖而用普通 `pip install llama-cpp-python` 覆盖它。该 wheel 的选择方法见[上游安装说明](https://github.com/JamePeng/llama-cpp-python#installation)；另可参考 [ComfyUI-QwenVL 的视觉版安装指南](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md)。

完成后重启 ComfyUI，并在节点菜单的 **Qwen/GGUF** 分类中找到两个节点。

## Qwen3.8 自动下载预设

在推理节点的 `model` 下拉框选择以 `Download | Unsloth Qwen3.8` 开头的选项，**第一次执行时**才会下载。四个预设均来自 [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main)：

| 预设量化 | 主模型下载量 | 说明 |
| --- | ---: | --- |
| `IQ2_S` | 8.37 GB | 占用较低，适合先试用 |
| `IQ3_XXS` | 10.93 GB | 中等体积 |
| `IQ3_S` | 12.04 GB | 更高的量化精度 |
| `Q3_K_XL` | 13.15 GB | 四项中体积最大 |

节点同时下载同一仓库的 `mmproj-F16.gguf`（0.93 GB），以便直接使用视觉推理和思考开关；`mmproj` 下拉框的选择会被预设自动匹配覆盖。文件保存到 `ComfyUI/models/LLM/unsloth/Qwen3.8-27B-GGUF/`；已有完整文件会直接复用，下载中断后再次执行可继续。预设固定在仓库提交 `4ca7207`，使主模型与 mmproj 版本一致。

这些是下载大小，不是推理所需显存。还需为 mmproj、上下文缓存和 ComfyUI 其他节点留出空间；显存不足时请选更小量化或减少 `gpu_layers`。下载进度显示在 ComfyUI 控制台。未选择下载预设时不会访问 Hugging Face，本地 GGUF 用法保持不变。

## 快速使用

1. 添加 **Qwen 3.5+ GGUF Inference**，选择本地 `model` 或下载预设；本地模型有图像或视频输入时还要选择匹配的 `mmproj`。
2. 填写 `user_prompt`，按需修改 `preset_prompt` 与 `system_prompt`。预设内容会放在用户提示词之前；`Empty - Nothing` 不追加预设。
3. 按输入类型连线。需要更多选项时，添加 **Qwen GGUF Parameters** 并连接到 `parameters`。

| 你的数据 | 连接方式 | 建议模式 |
| --- | --- | --- |
| 纯文本 | 不连接媒体输入 | 默认即可 |
| 单图或普通图片批次 | `image: IMAGE` | `auto` 或 `images` |
| 图片形式的视频帧批次 | `image: IMAGE` | 手动选 `video`；已知帧率也可填 `video_fps` 后用 `auto` |
| ComfyUI 原生视频 | `video: VIDEO` | `auto` 或 `video`；帧率自动读取 |

`image` 和 `video` 只能连接一个。`IMAGE` 批次本身不携带“这是视频”的标记或帧率，因此未知帧率的视频帧批次需要手动选 `video` 模式。原生 `VIDEO` 使用自身帧率，忽略 `video_fps`。

## 视频与批次如何处理

- `max_frames` 默认 24。输入超过上限时，节点会从**整段序列均匀抽样**，不会只取前 24 帧；未抽中的帧不参与推理。该上限也适用于普通图片批次。
- `max_size` 默认 256，限制每帧最长边。小字或 OCR 可适当提高，但显存与处理时间也会增加；0 表示不缩放。
- 原生 `VIDEO` 的帧率会用于时间戳；图片帧批次可在 Parameters 中填写 `video_fps`。未知帧率时只标注帧序号。
- 音轨不参与推理。当前实现把视频整理成有序图片帧发送给视觉处理器，并不发送原生视频媒体消息。
- ComfyUI 的原生视频对象可能在抽样前先解码整段视频。长视频建议先在上游裁剪或分段。

| `inference_mode` | 行为 |
| --- | --- |
| `images` | 将抽中的图片放进一次多模态请求 |
| `video` | 将抽中的帧按时间顺序放进一次请求，并添加视频分析指令 |
| `one by one` | 每帧单独分析，再用一次文本请求汇总；N 帧通常会调用模型 N+1 次 |
| `auto` | 最多 8 帧且估计图像 token 不超过上下文 60% 时，按输入类型选择 `images` 或 `video`；否则选择 `one by one` |

`auto` 的 token 估计是启发式规则，不能保证每个 GGUF 都适合当前上下文。视频帧较多而又希望减少模型调用次数时，可手动选择 `video`，并调整 `max_frames`、`max_size` 与 `context_size`。

## 节点与参数

**推理节点**保留常用选项：`attention_mode`（llama.cpp Flash Attention 的 `auto/on/off`）、`context_size`、`gpu_layers`、`keep_model_loaded` 和 `seed`。`gpu_layers=0` 表示不请求模型层 GPU 卸载；量化等级由所选 GGUF 文件决定。关闭 `keep_model_loaded` 会在本次推理后释放模型。

**Parameters 节点**可选；不连接时使用内置默认值。一个参数节点可以连接多个推理节点。

| 参数 | 作用 |
| --- | --- |
| `enable_thinking`、`reasoning_budget` | 启用思考模板并设置预算；预算 `-1` 不限制。当前处理器开启思考时需要 mmproj |
| `max_tokens`、`temperature`、`top_k`、`top_p`、`min_p`、`typical_p` | 输出长度和采样策略 |
| `repeat_penalty`、`frequency_penalty`、`presence_penalty` | 重复与话题惩罚 |
| `mirostat_mode`、`mirostat_eta`、`mirostat_tau` | 自适应采样；模式 0 关闭 |
| `inference_mode`、`max_frames`、`max_size`、`video_fps` | 批次/视频处理，见上文 |
| `image_max_tokens` | 单图视觉 token 上限；`-1` 使用模型元数据默认值 |
| `n_batch`、`n_threads` | llama.cpp 批次与 CPU 线程；线程数 0 用后端默认值 |
| `mtp_draft_tokens`、`mtp_draft_p_min` | MTP 推测解码；草稿 token 数 0 关闭，需兼容的模型及后端 |

`stats_json` 返回 token 用量、所选模式、总帧数、抽样帧数、模型调用次数，以及可用的 MTP 统计。`gpu_layers_effective` 表示节点传给后端的 GPU 层数参数，并不是显存占用量或实际卸载层数的独立测量。

## 常见问题

**找不到 `Qwen35ChatHandler`**：确认安装的是带该处理器的 wheel，且安装命令使用的是 ComfyUI 自己的 Python；可运行上面的导入检查。

**自动下载失败**：检查 Hugging Face 网络连接和磁盘剩余空间，然后重新执行节点。已完成的文件会复用；未完成的下载由 Hugging Face Hub 继续处理。也可以在仓库页面手动下载相同文件并放入上述目录。

**GPU 利用率低、CPU 很忙**：检查 wheel 是否包含对应 GPU 后端，确认 `gpu_layers` 大于 0。`stats_json` 中该参数为 0 时，节点没有请求模型层 GPU 卸载。模型超过可用显存时，可换更小的 GGUF、降低上下文或释放其他显存占用。

**视频很慢**：`auto` 对较多帧可能选择 `one by one`，导致多次模型调用。尝试 `video` 模式，或降低 `max_frames`。首次运行还包含模型加载时间。

**输出内容看不到视频中的短暂事件**：提高 `max_frames`，或把长视频切成多个短片段分别处理。均匀抽样无法观察未选中的帧。

## 项目来源

实现参考了 [ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) 与 [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL)，仅保留本项目需要的 GGUF 工作流。MTP 细节可查阅 [llama.cpp 推测解码文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)。
