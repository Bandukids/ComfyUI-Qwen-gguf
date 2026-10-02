# 参数手册

[返回首页](../README.md) · [使用教程](usage.md) · [常见问题](troubleshooting.md)

以下名称和默认值对应当前代码。未连接 Parameters 时，使用相同内置默认值。

## 主推理节点

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `model` | 本地列表或下载选项 | 主模型 GGUF，不能选择 mmproj 或单独的 draft 模型替代 |
| `mmproj` | `None` | 匹配的视觉投影；图片/视频和开启思考时必选；下载预设自动匹配 |
| `preset_prompt` | `Normal - Describe` | 追加到用户提示词前面的任务模板 |
| `user_prompt` | `Describe this image.` | 本次请求的任务、语言、格式和限制 |
| `system_prompt` | `You are a helpful assistant.` | 角色、语言、回答规则等系统消息 |
| `attention_mode` | `auto` | llama.cpp Flash Attention 的 `auto / on / off`，由后端支持情况决定 |
| `context_size` | `8192` | 输入、视觉 token、思考与答案可用的上下文容量；可选512–262144，不代表所有模型都能使用上限 |
| `gpu_layers` | `99` | 向后端请求放到 GPU 的模型层数；0 为不请求模型层卸载，允许0–999 |
| `keep_model_loaded` | `true` | 执行后保留权重，减少下次加载时间；false 执行后关闭模型 |
| `seed` | `0` | 采样随机种子；是否每次变化还取决于控件的生成后行为 |
| `enable_thinking` | `false` | 主界面思考总开关，优先于旧 Parameters 设置 |
| `thinking_level` | `auto` | 思考强度，见[强度说明](usage.md#思考模式与强度) |

### GPU Layers 与显存

`gpu_layers` 指模型层放到 GPU 执行的数量，即常说的模型 offload。默认99的意图是对层数少于该值的模型请求尽可能多的层；它不是99%显存，也不是99 GB。后端按模型实际层数和构建决定怎么加载。

`gpu_layers=0` 不请求主模型层 GPU 卸载，也不等于整条 ComfyUI 工作流完全不使用 GPU。降低层数可以减少显存需求，但可能让 CPU 参与更多计算、降低速度。若后端没有 GPU 能力，节点会把有效请求设为0，并在控制台说明。

量化选择通过 GGUF 文件完成，例如 Q4 与 Q8，不另设“quantization”控件。更大上下文会占用额外缓存空间；其他生图模型与 GGUF 常驻也会争用同一张显卡。

### Attention Mode

这里只控制 llama.cpp 的 Flash Attention。`sage`、`sdpa`、`flash_attention_2` 是其他后端常见的选项，不能直接当成本节点的可选项。后端报 Flash Attention 不兼容时可尝试 `off`，平时先使用 `auto`。

## Parameters 节点

| 参数 | 默认值 | 作用与注意事项 |
| --- | --- | --- |
| `enable_thinking` | `false` | 为旧工作流保留；主节点已提供开关时以主节点为准 |
| `reasoning_budget` | `-1` | 主节点 `thinking_level=custom` 时使用；0关闭，正数限制思考 token，-1不设思考预算 |
| `inference_mode` | `auto` | `auto / one by one / images / video` |
| `max_frames` | `24` | 每次最多选中多少张图片/帧，1–1024；超过时均匀抽样 |
| `max_size` | `256` | 图片最长边限制，0–4096；0不缩放，小图不会被放大 |
| `video_fps` | `0.0` | IMAGE帧批次的帧率；0不生成时间戳；原生VIDEO使用自身帧率 |
| `max_tokens` | `1024` | 无思考时为总输出上限；有限思考预算时为答案预留空间，见下文 |
| `top_k` | `30` | 候选 token 数；0按后端行为关闭此筛选 |
| `top_p` | `0.90` | 按累计概率筛选候选 |
| `min_p` | `0.05` | 相对最高概率 token 的概率阈值 |
| `typical_p` | `1.0` | 典型采样阈值；1使用默认放宽状态 |
| `temperature` | `0.80` | 控制采样随机性，控件范围0–2 |
| `repeat_penalty` | `1.0` | 重复惩罚；1表示此项不额外惩罚 |
| `frequency_penalty` | `0.0` | 按 token 出现频率调整概率 |
| `presence_penalty` | `0.0` | 调整已出现 token 的概率 |
| `mirostat_mode` | `0` | 自适应采样，0关闭，其他模式取决于后端 |
| `mirostat_eta` | `0.10` | Mirostat 学习率 |
| `mirostat_tau` | `5.0` | Mirostat 目标惊讶度 |
| `image_max_tokens` | `-1` | 视觉 token 上限提示；-1使用模型元数据默认值，效果由处理器与模型决定 |
| `n_batch` | `512` | 后端输入处理批次参数，32–8192；不是图片数量 |
| `n_threads` | `0` | CPU线程参数，0由后端选择，最大128；更多不一定更快 |
| `mtp_draft_tokens` | `0` | MTP草稿长度，0关闭，最大32，且必须小于 `n_batch` |
| `mtp_draft_p_min` | `0.0` | MTP草稿概率阈值，0–1 |

采样参数会相互作用。思考强度不会自动覆盖温度、惩罚等设置；选择强度和调整采样是两个不同控制维度。

### max_tokens 与思考预算

有限思考预算下，总上限为 `max_tokens + 预算 + 64`，再根据实际输入 token 和上下文缩减预算。64是结束思考标签、结论引导等的收尾余量。模型提前结束思考时，答案可以使用剩余空间，所以 `max_tokens` 不是最终答案长度的独立硬上限。

`custom` 配 `reasoning_budget=-1` 保持不限制思考预算的行为，但总输出仍受 `max_tokens` 限制。若想避免思考耗尽全部输出空间，使用有限预算档位。输入占满上下文时，增大输出参数并不能解决输入容量不足。

### MTP 怎么开启

需同时具备包含兼容 MTP 层的主模型和支持该机制的后端。把 `mtp_draft_tokens` 从0改成较小正数，再观察控制台和 `stats_json.mtp`。独立 draft GGUF 并非本节点的主模型输入，当前没有单独的 draft 模型路径控件。

节点会请求后端的 `DRAFT_MTP` 配置，但是否可用、是否加速取决于具体模型和构建。初始化失败先恢复为0。[llama.cpp 推测解码说明](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)

## 如何看 stats_json

| 字段 | 用途 |
| --- | --- |
| `usage` | 后端返回的 token 用量；逐帧模式累计各次调用 |
| `inference_mode` | 实际使用的模式，可能为 `one_by_one` |
| `total_frames` / `selected_frames` | 输入数量与抽样后数量 |
| `model_calls` | 本次执行调用模型的次数 |
| `gpu_layers_effective` | 节点传给后端的有效层数请求，不是实际卸载层数的独立测量 |
| `mtp` | 后端可提供的 MTP 统计，关闭时通常为空 |
| `thinking.enabled` / `level` | 思考开关与档位 |
| `thinking.budget_requested` | 请求的预算；实际值可能因上下文被缩减 |
| `thinking.calls` | 每次调用的阶段、输入 token、实际预算、生成上限及模板预填状态 |

`thinking.calls` 中 `stage=frame` 是单帧观察，`stage=final` 是最终回答。后端的不同版本可能提供不同的 `usage` 或 MTP 子字段；不要将空统计直接解读为0耗时或0显存。
