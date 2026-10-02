# 常见问题与排错

[返回首页](../README.md) · [安装指南](installation.md) · [使用教程](usage.md) · [参数手册](parameters.md)

先看错误发生在哪一阶段：节点导入 → 文件选择/下载 → 主模型加载 → 视觉初始化 → 推理 → 下游模型执行。ComfyUI 错误报告末尾的异常和错误附近控制台日志通常最有用。

## 节点没有出现在菜单中

确认文件位于 `custom_nodes/ComfyUI-Qwen-gguf/__init__.py`，安装通用依赖和后端时使用了 ComfyUI 自己的 Python。重启后查看启动日志，搜索 `ComfyUI-Qwen-gguf`、`IMPORT FAILED`、`ModuleNotFoundError`。

`requirements.txt` 只安装 Pillow 与 Hugging Face Hub；后端需要另装。具体路径与检查命令见[安装指南](installation.md)。

## 找不到 Qwen35ChatHandler / GenericMTMDChatHandler

当前 Python 环境中的 `llama-cpp-python` 构建没有所需接口，或依赖装进了另一个 Python。按[安装验证命令](installation.md#第三步安装视觉版-llama-cpp-python)检查版本及两个处理器导入，选择支持视觉接口的上游 wheel。

若出现 `unexpected keyword argument`，记录具体参数名：可能是后端缺少思考预算、Flash Attention 或其他接口。导入成功并不保证所有较新模型和所有接口都兼容。

## IMAGE input requires the matching mmproj GGUF file

已经连接图片或视频，但 `mmproj=None`。选择主模型发布者指定的配套 mmproj；下载预设会自动匹配。不要为了消除报错随便选择另一规模模型的 mmproj。

## Failed to load mtmd context

这表示视觉处理器创建多模态上下文失败，尚未开始回答。按顺序检查：

1. **主模型与 mmproj 是否配套。** 例如 Qwen3.5 **9B** + Qwen3.8 **27B** mmproj 是错误配对，改成发布者指定的9B文件。
2. **文件是否完整。** 确认下载完成，文件不是下载页面的HTML，也没有缺失分片；必要时按发布者校验值检查。
3. **后端是否支持该模型。** 确认视觉版后端及所需聊天模板支持；主模型名中保留系列标识。
4. **错误附近是否有显存/分配失败日志。** 有图像输入时，还要容纳视觉模型与缓存；先关闭其他显存占用，减小主模型GPU层数或上下文。

这个异常是上层统一错误，单凭这一行不能断定是文件损坏或显存不足。提供底层日志能更准确定位。

## 很慢，CPU负载高，GPU利用率低

| 检查项 | 处理方法 |
| --- | --- |
| 首次下载或加载 | 与后续推理分开观察；首次耗时包含这两个阶段 |
| 后端是否具备GPU能力 | 看控制台CUDA后端/offload信息；PyTorch能用GPU不代表llama后端也能用 |
| `gpu_layers` | 大于0才请求主模型层GPU卸载；不足显存时部分层需CPU执行 |
| `stats_json.gpu_layers_effective=0` | 这次没有请求GPU层，检查设置与后端检测信息 |
| `model_calls` 很大 | 通常是逐帧推理；减少帧数，或确认上下文足够后手动选择 `video/images` |
| 思考预算过大 | 关闭思考或改低档，比较实际生成 token 数 |
| 图片过大 | 减小 `max_size` 与帧数；OCR等任务再按需要增加 |
| 多模型共用GPU | 释放其他模型，或换更小GGUF、减小上下文 |

CPU也会参与图片转换、数据处理和后端调度，因此CPU有负载并不能单独证明完全没走GPU。`gpu_layers_effective` 是请求值，实际卸载情况还应结合后端日志。

## 显存不足 / CUDA分配失败

先用较小模型、单图、默认尺寸和适当上下文跑通。再逐项增加；模型文件大小不是全部显存开销。降低 `gpu_layers` 可减少显存需求，但可能降低速度。

`keep_model_loaded=true` 让GGUF权重常驻，方便连续推理；它不会自动被ComfyUI的其他模型管理器卸载。若后续要跑较大的生图模型，可以把它设为false，让节点执行后关闭GGUF。

## 视频只有部分动作 / 时间戳不对

超过 `max_frames` 会均匀抽样，短暂事件可能没被抽到。提高帧数、缩短片段或在上游分段。`IMAGE` 视频帧应填写实际帧批次的帧率，上游重采样后不要仍填原视频fps。

未知帧率的图片序列用 `video` 模式、`video_fps=0`，输出使用帧序号。原生VIDEO会读取自己的帧率。音轨不参与推理。

## 输出还是很长 / 思考内容出现了

确认文本显示连接的是 `response`，而不是 `reasoning`。`response` 分离标准 `<think>...</think>` 思考块；未使用这些标记的普通分析性文字仍可能属于模型最终回答，需要在提示词中约束格式。

不需要思考时，将主节点 `enable_thinking=false`。需要短答案时同时写清“只输出一段/三句话/一个词”等要求，再按任务调整 `max_tokens`。有限预算下它是答案预留空间，不是独立答案硬上限。

## reached max_tokens during thinking / 没有最终答案

自定义无限预算 `reasoning_budget=-1` 可能在生成答案前耗尽总输出空间。先改成 `auto/low/medium` 或有限自定义预算，或者关闭思考。查看 `stats_json.thinking.calls` 的实际输入、预算和生成上限。

输入已接近占满上下文时，要减少帧数/尺寸，或增加 `context_size`；仅增加 `max_tokens` 不足以解决输入过长。

## 下载失败或下拉框没有模型

下载失败先检查Hugging Face连接和剩余磁盘空间，重新执行可继续未完成下载。也可手动放入对应目录，见[下载说明](installation.md#自动下载模型)。

本地文件必须是 `models/LLM/` 下的GGUF；mmproj文件名需包含 `mmproj`。新文件没有出现时重启/刷新列表。`Select a local model or download preset` 是无本地模型时的提示，占位项不能推理。

## MTP失败 / unused tensor ... ignoring

MTP初始化失败先设 `mtp_draft_tokens=0`，核对主模型包含兼容MTP层、后端支持 `DRAFT_MTP`。独立draft文件不能作为主模型直接替代。

`unused tensor ... ignoring` 可能是未启用MTP等可选层被忽略，不一定是执行失败。应看后面是否出现异常，不能只据这条日志判定模型损坏。

## 工作流缺少节点 / 更新后没有新控件

`missing_node_type` 或 `has no class_type` 表示某个工作流节点缺失或结构有问题，应先检查报告中的节点ID与节点类型。这与视觉模型加载失败是不同阶段。

更新代码后重启ComfyUI，刷新浏览器。旧工作流缺少新字段或控件排列不正常时，先保存备份，再新建对应节点、填写参数并重新连接。

## 如何提交有用的Issue

在[GitHub Issues](https://github.com/Bandukids/ComfyUI-Qwen-gguf/issues)附上：

- ComfyUI、Python、`llama-cpp-python`版本，以及安装的wheel文件名。
- 操作系统、GPU型号与显存。
- 主模型和mmproj的完整文件名及下载仓库链接。
- 输入类型、帧数、`gpu_layers`、`context_size`、思考与Parameters设置。
- 异常堆栈及错误发生前后的控制台日志；能输出时附 `stats_json`。
- 单图最小工作流是否也能复现。

分享工作流或日志前检查API密钥、私人提示词和文件路径等内容。优先用不含隐私的示例图片复现。
