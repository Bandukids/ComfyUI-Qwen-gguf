# ComfyUI Qwen 3.5+ GGUF

一个精简的 ComfyUI 节点，用本地 GGUF 模型运行 Qwen 3.5 及后续版本的文本或图像推理。只有一个节点；不下载模型，不安装 Transformers，也不支持早期 Qwen 模型。

## 准备

1. 将本目录放进 `ComfyUI/custom_nodes/`，重启 ComfyUI。
2. 在 **ComfyUI 自己的 Python 环境**中安装带有 `Qwen35ChatHandler` 的 `llama-cpp-python`。推荐使用 [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)；可在 [Releases 页面](https://github.com/JamePeng/llama-cpp-python/releases)按操作系统、Python 版本和 GPU 后端选择预编译 wheel。该项目的 `0.3.49` 版本已验证提供所需接口。如果已安装近期版本的 `ComfyUI-QwenVL` 及其 GGUF 依赖，通常可以复用，不需要 `llama-server`。
3. 将主模型 `.gguf` 和**匹配该模型的**视觉投影文件 `mmproj*.gguf` 放进 `ComfyUI/models/LLM/` 的任意子目录（例如 `LLM/Qwen-VL/` 或 `LLM/GGUF/`）。重启或刷新节点后选择两个文件。纯文本推理时 `mmproj` 可以选 `None`。

节点还使用 ComfyUI 自带的 PyTorch 和 Pillow。安装 wheel 时请使用 ComfyUI 的 Python，而不是系统 Python。GPU 推理依赖带相应 GPU 后端的构建；若当前构建不支持 GPU，节点会自动使用 CPU 并在控制台提示。

## 节点用法

添加 **Qwen 3.5+ GGUF Inference**。填写系统提示词、可选的提示词预设和用户提示词，连接可选的 `IMAGE` 输入，输出为 `STRING`。预设指令会放在用户提示词之前；选 `None` 时只使用用户提示词。

ComfyUI 的 `IMAGE` 是 `[B,H,W,C]`；当 `B>1` 时，节点把整批图片按顺序放进**同一次请求**，适合比较图片或综合描述，输出一段文字。视频加载节点输出的图片帧批次也可以接到这个输入，但当前节点不会按时间戳处理视频。所有图片会无损编码为 PNG；大图或大批次可能需要提高 `context_size`。

基础控件包括 `max_tokens`、`temperature`、`seed`、`attention_mode` 和 `keep_model_loaded`。`seed` 支持 ComfyUI 的“生成后控制”选项。`attention_mode` 可选 `auto`、`disabled`、`enabled`，更改后会重新加载模型。关闭 `keep_model_loaded` 会在本次推理结束后释放模型。量化等级由选择的 GGUF 文件决定，因此没有单独的量化控件。

首次运行会加载模型，默认在后续运行复用同一模型；更换模型、mmproj、上下文、有效 GPU 层数或注意力模式会重新加载。ComfyUI 退出时会释放模型。

实现参考了 [ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) 和 [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL) 的 GGUF 加载与图像消息形式。当前节点直接调用已安装的 `llama-cpp-python`，无需运行独立服务。
