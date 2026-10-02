# ComfyUI Qwen GGUF

**在 ComfyUI 中使用 Qwen 3.5+ GGUF，完成文本问答、图片描述、多图比较、OCR 与视频帧分析。**

一个主推理节点，加一个可选的 Parameters 节点。推理在本机运行，无需启动 `llama-server`；可手动放置模型，也可在节点中选择下载预设。

[安装指南](docs/installation.md) · [使用教程](docs/usage.md) · [参数手册](docs/parameters.md) · [常见问题](docs/troubleshooting.md) · [English quick start](docs/english.md)

## 第一次使用，从这里开始

1. 按[安装指南](docs/installation.md)安装节点、`requirements.txt` 和视觉版 `llama-cpp-python`。三者都需要安装到 **ComfyUI 自己使用的 Python 环境**。
2. 重启 ComfyUI，在 **Qwen/GGUF** 分类添加 **Qwen 3.5+ GGUF Inference**。
3. 选择主模型和配套 mmproj；或选择 `Download | Unsloth Qwen3.8 ...` 预设，执行时自动下载配套文件。
4. 将加载图片节点的 `IMAGE` 输出连接到本节点的 `image`；填入 `user_prompt`，首次尝试保持 `enable_thinking=false`。
5. 将 `response` 连接到支持 `STRING` 的文本显示节点，然后执行工作流。

```text
图片加载节点 ── IMAGE ──▶ Qwen 3.5+ GGUF Inference ── response ──▶ 文本显示节点
                                  ▲
Qwen GGUF Parameters ── parameters ┘  （可选）
```

文本显示节点的名称取决于你安装的扩展；本项目输出字符串，不自带文本预览节点。先用单张图片跑通，再增加帧数、上下文或思考强度。完整步骤和提示词示例见[使用教程](docs/usage.md)。

## 主模型与 mmproj 必须配套

**`model` 是主模型，`mmproj` 是让主模型理解图片的视觉投影文件。mmproj 不能单独推理，也不是所有 Qwen 模型通用的配件。**

| 选择的主模型 | 应选择的 mmproj | 能否配套 |
| --- | --- | --- |
| Qwen3.5 **9B** GGUF | 发布者提供的 Qwen3.5 **9B** mmproj | 是，按模型仓库说明选择 |
| Qwen3.8 **27B** GGUF | 发布者提供的 Qwen3.8 **27B** mmproj | 是，按模型仓库说明选择 |
| Qwen3.5 **9B** GGUF | Qwen3.8 **27B** mmproj | **不匹配**，可能报 `Failed to load mtmd context` |
| 任意视觉模型 + 图片/视频输入 | `None` | 不可用，缺少视觉投影 |

模型系列、参数规模和发布者指定的配套关系都要核对；文件名仅是线索，最终以模型仓库说明为准。主模型的 `Q4`、`Q8`、`IQ3` 等量化名称不必与 mmproj 的 `F16/BF16` 一致。[文件摆放与配对实例](docs/installation.md#模型文件放在哪里)

## 可以做什么

| 功能 | 当前行为 |
| --- | --- |
| 文本 | 不连接媒体输入，进行问答或提示词改写 |
| 单图 / 图片批次 | 使用 `image: IMAGE`；可同时比较多张图 |
| 图片形式的视频 | 将有序帧批次接入 `image`，指定 `video` 模式或提供帧率 |
| 原生视频 | 使用 ComfyUI `video: VIDEO`，读取图像帧与帧率 |
| 推理模式 | `auto`、`images`、`video`、`one by one` |
| 思考控制 | 主节点开关 + `auto / low / medium / high / custom` 五档强度 |
| 调参 | 采样、GPU 层数、上下文、Flash Attention、可选 MTP |
| 输出 | `response` 最终答案、`reasoning` 思考过程、`stats_json` 调试统计 |

`image` 与 `video` 每次只能连接一个。视频按有序图片帧分析，音轨不参与推理；超过 `max_frames` 时从整段序列均匀抽样，默认上限 24 帧。`auto` 可能逐帧调用多次模型，详细规则见[批次与视频教程](docs/usage.md#视频与图片批次)。

## 自动下载选项

四个预设来自 [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main)。选择后首次**执行节点**时下载，配套 `mmproj-F16.gguf` 会自动匹配，无需另选。

| 量化 | 主模型下载量 | 加上 mmproj 后约需存储空间 |
| --- | ---: | ---: |
| `IQ2_S` | 8.37 GB | 9.30 GB |
| `IQ3_XXS` | 10.93 GB | 11.86 GB |
| `IQ3_S` | 12.04 GB | 12.97 GB |
| `Q3_K_XL` | 13.15 GB | 14.08 GB |

文件大小按十进制 GB 表示，**不是所需显存**；运行还需要视觉模型、上下文缓存与其他 ComfyUI 模型的空间。文件保存到 `ComfyUI/models/LLM/unsloth/Qwen3.8-27B-GGUF/`，完整文件会复用，中断后可重试继续下载。[下载与手动安装说明](docs/installation.md#自动下载模型)

## 安装与兼容性

| 组件 | 要求 |
| --- | --- |
| ComfyUI | 已有可运行的环境；PyTorch 由 ComfyUI 提供 |
| 普通依赖 | `requirements.txt`：Pillow、Hugging Face Hub |
| GGUF 后端 | 支持 Qwen 视觉、思考预算等接口的 [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) 构建，单独安装 |
| GPU | 选择与系统、Python 和 GPU 后端匹配的 wheel；CPU 也可运行，但通常较慢 |

下载 wheel 的入口：[llama-cpp-python Releases](https://github.com/JamePeng/llama-cpp-python/releases)。`requirements.txt` 不强制安装它，以免覆盖已有可用的 GPU 构建。

本项目面向 Qwen3.5 及后续 GGUF，不提供早期 Qwen 支持。Qwen3.5 使用 `Qwen35ChatHandler`，文件名包含 `Qwen3.8` 的模型使用 GGUF 聊天模板驱动的 `GenericMTMDChatHandler`；请保留名称中的系列标识。其他后续模型能否运行，仍取决于其 GGUF、mmproj 和后端兼容性。

已在 Windows、RTX 4070 Ti SUPER、视觉版 `llama-cpp-python 0.3.49` 与 Qwen3.5 9B 下实测文本、图片及低/中档思考预算。30 个单元测试覆盖批次、视频规则、预设下载、参数传递与预算控制；这不代表所有平台、量化或 MTP 组合都已实测。

## 遇到问题时

| 现象 | 先检查 |
| --- | --- |
| 节点不出现 / 导入失败 | 安装位置、ComfyUI Python、视觉后端导入检查 |
| `IMAGE input requires ... mmproj` | 图片/视频输入时，mmproj 是否为 `None` |
| `Failed to load mtmd context` | 主模型与 mmproj 是否配套，其次检查文件、后端和显存 |
| 很慢 / CPU 负载高 | GPU wheel、`gpu_layers`、模型是否装得下、是否逐帧推理 |
| 答案为空 / 思考耗尽输出 | 思考档位、`max_tokens` 和上下文空间 |
| 下游生图时显存不足 | GGUF 是否常驻；可关闭 `keep_model_loaded` |

完整步骤与提交 Issue 所需信息见[常见问题](docs/troubleshooting.md)。更新后需要重启 ComfyUI；旧工作流看不到新控件时，可新建节点并重新连接。

## 项目来源与反馈

实现参考 [ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) 与 [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL)，围绕 GGUF 精简工作流。

[反馈问题](https://github.com/Bandukids/ComfyUI-Qwen-gguf/issues)时，请附上具体主模型/mmproj 文件名、后端版本、关键参数和错误附近的控制台日志。
