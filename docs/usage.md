# 使用教程

[返回首页](../README.md) · [安装指南](installation.md) · [参数手册](parameters.md) · [常见问题](troubleshooting.md)

## 两个节点怎么分工

在节点菜单的 **Qwen/GGUF** 分类中添加：

| 节点名称 | 用途 |
| --- | --- |
| **Qwen 3.5+ GGUF Inference** | 选择模型、接入图片/视频、填写提示词、控制思考，输出文本 |
| **Qwen GGUF Parameters** | 选填的采样、帧数、尺寸与高级参数，通过 `parameters` 连接主节点 |

不连接 Parameters 也能执行，主节点会使用内置默认值。Parameters 不是模型加载节点；它只传递设置，可连接多个主节点。

## 跑通第一张图片

1. 添加图片加载节点，选择一张简单图片。
2. 添加主推理节点，选主模型和发布者指定的匹配 mmproj；下载预设会自动匹配。
3. 把图片节点的 `IMAGE` 输出接到主节点 `image`。`video` 暂不连接。
4. `preset_prompt` 选 `Normal - Describe`，`user_prompt` 填“请用中文简洁描述图片中的主体、环境和光线”。
5. `system_prompt` 可填“你是图像描述助手。只根据可见内容回答，无法确定的细节请说明”。
6. `enable_thinking=false`；其余参数先保持默认。
7. 把 `response` 接到任意支持 `STRING` 的文本显示节点，执行工作流。

```text
加载图片 ── IMAGE ──▶ [Qwen 3.5+ GGUF Inference] ── response ──▶ 文本显示
                                  ▲
                    可选：Parameters 节点
```

文本显示节点来自你已安装的扩展，名称可能不同。本项目不提供独立的文字预览节点；也可以把 `response` 接到接受 `STRING` 的后续节点或文本保存节点。

第一次执行包括下载（仅下载预设）和模型加载，通常比后续执行更慢。想确认是否走 GPU，可查看控制台的 CUDA/offload 信息，结合 `stats_json` 和显存占用判断。

## 纯文本问答与提示词改写

不连接 `image` 和 `video`，选择 `Empty - Nothing`，在 `user_prompt` 中直接写任务。例如：

```text
将下面的描述改写为英文图像生成提示词，只输出提示词：
黄昏时分，玻璃温室里摆放着一台白色钢琴，暖光穿过树叶。
```

纯文本且关闭思考时 mmproj 可为 `None`；开启思考需要选择匹配 mmproj。若没有媒体输入却仍使用描述图片类预设，模型可能误解任务，因此纯文本通常从 `Empty - Nothing` 开始。

## 提示词如何组合

`system_prompt` 是独立的系统消息，用来设定语言、角色和回答规则。用户消息由 **预设提示词 + user_prompt** 组成；预设放在用户文字之前，`Empty - Nothing` 不追加内容。

例如，要让结果更短，可以在 `user_prompt` 中写：

```text
请用中文描述这张图片。只写一段，最多三句话，不要分点，不添加开场白。
```

模型对字数或格式的遵循不是硬约束。`max_tokens` 是 token 数，不等于中文字数。关闭思考也不自动要求模型只写一句话；想要简短结果，需要同时明确任务格式。

### 全部预设

| 下拉选项 | 作用 |
| --- | --- |
| `Empty - Nothing` | 完全使用自定义用户提示词 |
| `Normal - Describe` | 准确描述图片 |
| `Prompt Style - Tags` | 简洁、逗号分隔的视觉标签 |
| `Prompt Style - Simple` | 简短的图像生成提示词 |
| `Prompt Style - Detailed` | 详细描述主体、环境、构图、光线与风格 |
| `Prompt Style - Extreme Detailed` | 进一步强调纹理、颜色、氛围等细节 |
| `Prompt Style - Cinematic` | 电影式取景、镜头视角、光线与情绪 |
| `Creative - Detailed Analysis` | 详细分析，区分观察与解释 |
| `Creative - Summarize Video` | 根据有序帧总结可见事件 |
| `Creative - Short Story` | 根据图像创作故事，虚构内容作为创作处理 |
| `Creative - Refine & Expand Prompt` | 结合可见细节扩写用户提示词 |
| `Vision - *Bounding Box` | 请求近似的归一化 `[x1,y1,x2,y2]` 框 |
| `Extract Text (OCR)` | 转写可读文字，保留原语言与换行 |
| `Compare Images` | 按图片顺序比较异同 |

预设是提示词模板，不是额外训练的模型功能。例如 Bounding Box 输出是文本中的近似位置，不是可直接连接检测节点的结构化框；OCR 效果取决于图片清晰度、尺寸与模型能力。

## 多张图片同时比较

将上游输出的 `IMAGE` 批次接入 `image`，连接 Parameters，设 `inference_mode=images`，再选 `Compare Images`。普通单张 `IMAGE` 同样可接入。

示例用户提示词：

```text
按输入顺序称为图片1、图片2。比较主体、构图和光线。
只列出三项最明显的差异，不推断看不见的内容。
```

`images` 会在一次请求中发送抽样后的图片。`one by one` 会分别观察，再汇总成一个答案，不能在同一次视觉请求中直接比较所有图。主节点最终仍输出一个 `response` 字符串，并非每张图片各输出一个独立字符串。

`max_frames` 也限制普通图片批次。若所有图片都必须参与，应把上限设为至少等于批次数量，同时确认上下文足够。

## 视频与图片批次

两种视频接法都支持，但每次只能连接 `image` 或 `video` 其中一个。

### 已解码为 IMAGE 的视频帧

```text
视频解码/帧加载节点 ── IMAGE批次 ──▶ 主节点 image
Parameters：inference_mode=video，video_fps=实际帧批次对应的帧率
```

`IMAGE` 批次本身不携带“这是视频”的标记。如果不知道帧率，手动选 `video` 并保持 `video_fps=0`，节点按帧序号描述。知道帧率时可填写 `video_fps`，再用 `auto`，节点会识别为视频序列并生成时间戳。

**填写的是输入帧批次的实际帧率。** 上游若把 30 fps 的原视频重采样为 2 fps 的帧批次，应填 2，而不是 30，否则时间戳会不准确。

### ComfyUI 原生 VIDEO

```text
输出原生 VIDEO 的加载节点 ── VIDEO ──▶ 主节点 video
```

节点读取 VIDEO 的图像帧和帧率，忽略 Parameters 的 `video_fps`。只有 `VIDEO` 类型可接此接口，视频路径字符串不能直接接入。若加载器只输出 `IMAGE`，使用上一种接法即可。

### 模式选择

| `inference_mode` | 执行方式 | 适用情况 |
| --- | --- | --- |
| `images` | 一次请求发送全部选中图片 | 多图比较、静态图片批次 |
| `video` | 一次请求发送有序帧，增加视频分析指令 | 希望整体分析视频片段 |
| `one by one` | 每帧一次观察，再一次汇总 | 帧数较多，单次上下文放不下 |
| `auto` | 按帧数与视觉 token 估算自动选择 | 初次使用或不确定模式 |

`auto` 在选中帧数不超过 8、估计图像 token 不超过上下文 60% 时，按是否有视频标记选择 `images` 或 `video`；否则选择 `one by one`。默认按单图 512 token 估算，正数 `image_max_tokens` 会参与估算。这是启发式规则，实际视觉 token 数仍由模型决定。

因此，选 24 帧后 `auto` 通常会进入逐帧流程；24 帧一般是 24 次观察 + 1 次汇总。想减少调用次数，可尝试手动 `video`，并确保上下文能够容纳这些帧。

### 超过24帧怎么办

默认 `max_frames=24`。超过上限时，从**整段序列均匀抽样**，不是只读开头。例如 120 帧会选出分布在整段片段中的 24 帧；未选中的帧不参与推理。

`max_size=256` 限制最长边，按比例缩小大图。小字、OCR、细节识别可调大；`0` 表示不缩放。增大尺寸或帧数都可能增加耗时、视觉 token 和显存需求。

短暂事件可能出现在未选帧之间。重要动作可提高抽样上限，或上游裁剪为更短的片段。原生 VIDEO 可能在抽样前先解码整段视频，长视频建议先分段。

当前视频推理只分析图像帧，不分析音轨，也不向后端发送原生视频媒体消息。时间戳是帮助模型理解顺序的文本信息，不保证事件发生时间达到逐帧精度。

## 思考模式与强度

主节点 `enable_thinking` 默认关闭，并优先于 Parameters 中保留的旧开关。开启后 `thinking_level` 生效：

| 选项 | 预算 |
| --- | --- |
| `auto` | `max_tokens` 的一半，限制在 128–1024；默认为 512 |
| `low` | 256 token |
| `medium` | 1024 token |
| `high` | 4096 token |
| `custom` | 采用 Parameters 的 `reasoning_budget`；`0` 关闭，`-1` 不限制思考预算 |

有限预算通过后端采样器控制，模板预填了 `<think>` 时也能计数。预算耗尽后插入结论引导、结束第一段思考，再继续生成答案。更高强度允许更长推理，但不保证每个任务都更准确。预算收尾机制参考[后端预算采样器实现](https://github.com/JamePeng/llama-cpp-python/blob/main/llama_cpp/_internals.py)。

有限预算下，`max_tokens` 用于为最终答案预留空间，总上限是 `max_tokens + 思考预算 + 64`，受实际上下文余量限制。例如默认答案空间 1024、低档 256，总上限为 1344。提前结束思考后，答案可使用剩余空间，所以它不是最终答案长度的独立硬上限。

上下文紧张时先缩减思考预算。`one by one` 的每帧最多思考 64 token，最终汇总使用所选强度。`custom=-1` 不受思考预算限制，但总输出仍受原来的 `max_tokens` 限制，可能在生成答案前耗尽。

修改强度不会重新加载权重；每次独立调用清理推理上下文，以避免当前混合架构后端重复提示词的缓存问题。切换思考总开关等模型配置可能需要重新加载。旧工作流未传入 `thinking_level` 时继续使用原 Parameters 预算。

## 输出接口如何使用

| 输出 | 类型 | 用途 |
| --- | --- | --- |
| `response` | `STRING` | 最终答案；接文本显示、保存或后续提示词节点 |
| `reasoning` | `STRING` | 单独的思考文本；未产生思考时为空 |
| `stats_json` | `STRING` | JSON 调试信息；可接另一个文本显示节点 |

只想看结果时，连接 `response`。`reasoning` 是最终一次模型调用的思考，不是逐帧观察的全部思考合集；预算强制收尾的引导文字可能出现在这里。

`stats_json` 可重点看：`inference_mode`、`selected_frames`、`model_calls`、`usage`、`gpu_layers_effective`，以及 `thinking.calls` 中实际的 `prompt_tokens`、`reasoning_budget`、`max_tokens`。这些统计帮助解释“为什么调用很多次”“高档为什么被缩减”。

## 常用设置组合

| 任务 | 可从这些设置开始 |
| --- | --- |
| 单图描述、标签 | 关闭思考，`images/auto`，最长边256，明确要求简短格式 |
| OCR | `Extract Text (OCR)`，适当提高 `max_size`，不要一次塞太多页 |
| 多图比较 | `Compare Images` + `images`，让 `max_frames` 覆盖全部待比较图片 |
| 短视频整体描述 | `video`，先少量帧验证，再增加帧数或上下文 |
| 复杂图像分析 | 开启思考，从 `auto/medium` 试起，查看实际预算与答案 |
| GGUF输出交给生图模型 | 按显存情况关闭 `keep_model_loaded`，避免模型权重同时常驻 |

这些是工作流起点，不是所有模型的最佳配置。一次修改一个主要变量，结合 `stats_json` 对比效果，通常更容易找到速度和质量之间合适的设置。
