# ComfyUI Qwen 3.5+ GGUF

一个精简的 ComfyUI 扩展，用本地 GGUF 模型运行 Qwen 3.5 及后续版本的文本、图片批次或视频帧推理。一个推理节点可独立使用，也可连接一个参数节点调节高级设置；不下载模型，不安装 Transformers，也不支持早期 Qwen 模型。

## 准备

1. 将本目录放进 `ComfyUI/custom_nodes/`，重启 ComfyUI。
2. 在 **ComfyUI 自己的 Python 环境**中安装带有 `Qwen35ChatHandler` 的 `llama-cpp-python`。推荐使用 [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)；可在 [Releases 页面](https://github.com/JamePeng/llama-cpp-python/releases)按操作系统、Python 版本和 GPU 后端选择预编译 wheel。该项目的 `0.3.49` 版本已验证提供所需接口。如果已安装近期版本的 `ComfyUI-QwenVL` 及其 GGUF 依赖，通常可以复用，不需要 `llama-server`。
3. 将主模型 `.gguf` 和**匹配该模型的**视觉投影文件 `mmproj*.gguf` 放进 `ComfyUI/models/LLM/` 的任意子目录（例如 `LLM/Qwen-VL/` 或 `LLM/GGUF/`）。重启或刷新节点后选择两个文件。纯文本推理时 `mmproj` 可以选 `None`。

节点还使用 ComfyUI 自带的 PyTorch 和 Pillow。安装 wheel 时请使用 ComfyUI 的 Python，而不是系统 Python。GPU 推理依赖带相应 GPU 后端的构建；节点会在检测前注册动态 ggml 后端。若当前构建仍不支持 GPU，节点会自动使用 CPU 并在控制台提示。可在 `stats_json` 的 `gpu_layers_effective` 查看本次模型实际采用的 GPU 层数；0 表示没有模型层卸载到 GPU。

## 节点用法

添加 **Qwen 3.5+ GGUF Inference**。`image` 和 `video` 两个可选输入现在都接受 ComfyUI 的 `IMAGE` 类型，但只能连接其中一个。普通图片或图片批次接 `image`；视频节点输出的有序帧批次接 `video`。其他输入包括提示词预设、用户提示词和系统提示词。输出为 `response`、`reasoning` 和 `stats_json`。预设指令会放在用户提示词之前；选 `Empty - Nothing` 时只使用用户提示词。`stats_json` 包含 token 用量、实际推理模式、取样帧数、模型调用次数，以及启用 MTP 时可用的推测解码统计。

ComfyUI 的 `IMAGE` 是 `[B,H,W,C]`，图片批次保留顺序。`video` 输入把批次视为按时间排列的视频帧，均匀选取最多 `max_frames` 帧；例如 240 帧且 `max_frames=24` 时，会从全片抽取 24 帧，而不是只取开头 24 帧。未抽中的帧不参与推理，`stats_json` 会给出 `total_frames` 和 `selected_frames`。`IMAGE` 批次不包含帧率；如需时间戳，可在 Parameters 节点填写 `video_fps`，默认 0 时只标注帧序号。需要更密集的观察时可提高上限，或先将视频分段推理。帧会先按 `max_size` 限制最长边，再编码为 PNG。音轨不参与推理；这套 Qwen3.5 处理器目前不接受原生视频媒体消息，因此这里使用有序帧图像。旧工作流若连接了原生 `VIDEO`，需改接视频解码节点输出的 `IMAGE` 帧批次。

推理节点保留 `seed`、`attention_mode`、`context_size`、`gpu_layers` 和 `keep_model_loaded`。`seed` 支持 ComfyUI 的“生成后控制”选项。`attention_mode` 控制 llama.cpp 的 Flash Attention，仅有 `auto`、`on`、`off`；它不对应 PyTorch 的 SageAttention、FlashAttention 2 或 SDPA。关闭 `keep_model_loaded` 会在本次推理结束后释放模型。量化等级由选择的 GGUF 文件决定，因此没有单独的量化控件。

预设列表参考了 QwenVL 的描述、标签、简单/详细/电影化提示词，以及详细分析、视频帧总结、短故事、提示词扩写和目标框选项；另保留 OCR 与多图比较。自定义用户提示词可以进一步限定输出语言与格式。

## Parameters 节点

需要调参时，添加 **Qwen GGUF Parameters**，把其 `parameters` 输出连接到推理节点的可选 `parameters` 输入。不连接时使用内置默认值。一个参数节点也可以连接多个推理节点。`enable_thinking` 与 `reasoning_budget` 位于参数节点顶部。

从旧版 `Qwen 3.5+ GGUF Inference (Advanced)` 工作流升级时，请用当前推理节点替换旧高级节点，并把原来的高级参数填入 Parameters 节点。

| 参数 | 用途 |
| --- | --- |
| `max_tokens`、`temperature`、`top_p`、`top_k`、`min_p`、`typical_p` | 控制输出长度和采样；`typical_p=1` 关闭典型采样限制。 |
| `repeat_penalty` | 调整重复惩罚；1.0 表示不施加。 |
| `frequency_penalty`、`presence_penalty` | 按出现次数或是否出现抑制重复内容。 |
| `mirostat_mode`、`mirostat_tau`、`mirostat_eta` | 自适应采样器：0 关闭，1/2 分别为两个版本；关闭时后两项不起作用。 |
| `enable_thinking`、`reasoning_budget` | 启用 Qwen 3.5 的思考模板，并可限制首个思考块的 token 数；`-1` 不限制。当前实现通过视觉处理器设置模板，因此开启思考时需要选择匹配的 `mmproj`。 |
| `inference_mode` | `images` 把选出的图片放进一次请求；`video` 把选出的帧作为按时间排序的视频序列放进一次请求，并附加视频分析指令；`one by one` 先逐帧分析，再用一次文本请求汇总。`auto` 在帧数不超过 8 且估计图像 token 不超过上下文 60% 时，按输入类型选 `images` 或 `video`，否则选 `one by one`。估计值不能保证一定适合当前模型，可手动切换。`video` 也使用图像帧，不发送原生视频媒体消息。 |
| `max_frames`、`max_size` | 分别限制均匀选取的帧/图片数和图像最长边；默认 24 帧、256 像素。超出上限的输入会均匀抽样，适用于所有推理模式。`max_size=0` 保留原尺寸。OCR 或小字识别可提高尺寸和上下文。 |
| `video_fps` | `video` 输入的帧率；默认 0 表示未知，只显示帧序号。设置实际帧率后，提示词中会增加相对时间戳。 |
| `image_max_tokens` | 视觉处理器的单图 token 上限；`-1` 使用模型元数据默认值。 |
| `n_batch`、`n_threads` | 逻辑批次和 CPU 线程数；线程数 0 使用后端默认值。物理微批次自动取 `min(512, n_batch)`。 |
| `mtp_draft_tokens`、`mtp_draft_p_min` | 0 关闭 MTP；大于 0 时启用内置 MTP 推测解码，分别设置最大草稿长度和草稿概率下限。只有包含 MTP 层的兼容 GGUF 才能使用。 |

MTP 是一种加速选项，不保证每个模型或硬件都更快；原理和 `draft-mtp` 模式见 [llama.cpp 推测解码文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)。修改加载类选项（上下文、批次、线程、视觉 token 限制、思考模式或 MTP）会重新加载模型；采样类选项不会。`num_beams` 和 `use_torch_compile` 属于 Transformers 节点的设置，这个 GGUF 后端不使用。MTP 与图像输入组合的实际支持取决于所安装的 llama-cpp-python 构建。

首次运行会加载模型，默认在后续运行复用同一模型；更换模型、mmproj 或加载类参数会重新加载。ComfyUI 退出时会释放模型。

实现参考了 [ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) 和 [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL) 的 GGUF 加载与图像消息形式。当前节点直接调用已安装的 `llama-cpp-python`，无需运行独立服务。
