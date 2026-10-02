# 安装指南

[返回首页](../README.md) · [使用教程](usage.md) · [常见问题](troubleshooting.md)

## 安装前准备

先确认 ComfyUI 本身可以运行。安装分为三部分：节点代码、`requirements.txt` 的普通依赖、视觉版 `llama-cpp-python` 后端。安装普通依赖不会自动安装后端，也不会下载模型。

节点在 ComfyUI 进程内加载 GGUF，无需启动 `llama-server`。模型放在本地后可以离线推理；下载预设需要访问 Hugging Face。[ComfyUI 官方自定义节点安装指南](https://docs.comfy.org/installation/install_custom_node)

## 第一步：定位 ComfyUI 和 Python

Windows 用户在 PowerShell 中设置两个变量，按实际安装位置修改。后续命令沿用这两个变量。

### Windows 便携版

```powershell
$ComfyRoot = 'C:\AI\ComfyUI_windows_portable\ComfyUI'
$ComfyPython = 'C:\AI\ComfyUI_windows_portable\python_embeded\python.exe'
```

标准便携版目录通常叫 `python_embeded`，拼写以本机目录为准。

### Windows 整合包 / 秋叶整合包

```powershell
$ComfyRoot = 'D:\AI Software\ComfyUI-aki-v3.2\ComfyUI'
$ComfyPython = 'D:\AI Software\ComfyUI-aki-v3.2\python\python.exe'
```

### Windows 虚拟环境

项目内 `.venv` 的示例；其他虚拟环境换成实际解释器路径：

```powershell
$ComfyRoot = 'C:\AI\ComfyUI'
$ComfyPython = 'C:\AI\ComfyUI\.venv\Scripts\python.exe'
```

确认目录、Python 版本及解释器：

```powershell
Test-Path -LiteralPath $ComfyRoot
Test-Path -LiteralPath $ComfyPython
& $ComfyPython -c "import sys; print(sys.executable); print(sys.version)"
```

两条路径检查应返回 `True`，解释器应与启动 ComfyUI 的环境一致。命令前的 `&` 用于执行变量中的程序路径，路径包含空格时也能正常运行。

## 第二步：安装节点与普通依赖

```powershell
git clone https://github.com/Bandukids/ComfyUI-Qwen-gguf.git "$ComfyRoot\custom_nodes\ComfyUI-Qwen-gguf"
& $ComfyPython -m pip install -r "$ComfyRoot\custom_nodes\ComfyUI-Qwen-gguf\requirements.txt"
```

已经安装过则无需再次 clone，见[更新节点](#更新节点)。没有 Git 时，可[下载 ZIP](https://github.com/Bandukids/ComfyUI-Qwen-gguf/archive/refs/heads/main.zip)，解压后命名为 `ComfyUI-Qwen-gguf`，放到 `custom_nodes`，再安装依赖。

正确结构是 `custom_nodes/ComfyUI-Qwen-gguf/__init__.py`，不要多嵌套一层文件夹。

### Linux / macOS / 已激活的虚拟环境

使用启动 ComfyUI 的同一个 Python。以下路径需替换为实际目录：

```bash
cd /path/to/ComfyUI
python -c "import sys; print(sys.executable); print(sys.version)"
git clone https://github.com/Bandukids/ComfyUI-Qwen-gguf.git custom_nodes/ComfyUI-Qwen-gguf
python -m pip install -r custom_nodes/ComfyUI-Qwen-gguf/requirements.txt
```

## 第三步：安装视觉版 llama-cpp-python

[后端项目](https://github.com/JamePeng/llama-cpp-python) · [wheel 下载页](https://github.com/JamePeng/llama-cpp-python/releases) · [上游安装说明](https://github.com/JamePeng/llama-cpp-python#installation)

在 Releases 的 Assets 中下载对应 wheel，核对这些条件：

| 条件 | 怎么选择 |
| --- | --- |
| 系统与架构 | Windows / Linux / macOS，以及 x64 / ARM 等 |
| Python 标签 | 如 `cp312` 对应 Python 3.12，`cp313` 对应 Python 3.13 |
| 加速后端 | NVIDIA 选择合适的 CUDA 构建；macOS 按上游说明选择 Metal；CPU 构建也需满足视觉接口要求 |
| 推理接口 | 包含 Qwen 视觉处理器，以及思考预算、Flash Attention 等本项目使用的接口 |

CUDA 构建的驱动和运行库要求按对应 release 说明核对。PyTorch 支持 CUDA，并不代表 `llama-cpp-python` 已有可用的 CUDA 后端。

Windows 安装下载好的 wheel，将占位文件名改为真实文件名：

```powershell
& $ComfyPython -m pip install 'C:\Downloads\<实际下载的wheel文件名>.whl'
& $ComfyPython -c "import llama_cpp; from llama_cpp.llama_chat_format import Qwen35ChatHandler; from llama_cpp.llama_multimodal import GenericMTMDChatHandler; print('llama-cpp-python:', llama_cpp.__version__); print('Vision handlers OK')"
& $ComfyPython -c "import inspect; from llama_cpp import Llama; print('Reasoning budget API:', 'reasoning_budget' in inspect.signature(Llama.create_completion).parameters)"
```

Linux/macOS 将 `& $ComfyPython` 换成对应环境的 `python`，安装相应 wheel，或按上游文档从源码构建。

本项目实测后端版本为视觉版 `0.3.49`，不表示所有较新版本都已验证。已有可用 GPU 后端时，无需仅为安装本节点而替换它。普通 `pip install llama-cpp-python` 可能装成其他构建，所以后端没有直接列进 `requirements.txt`。

安装后**重启 ComfyUI**，在 `Qwen/GGUF` 分类查找两个节点。没有出现时先看启动日志里的 `IMPORT FAILED`，按[排错指南](troubleshooting.md)处理。

## 模型文件放在哪里

```text
ComfyUI/
├─ custom_nodes/
│  └─ ComfyUI-Qwen-gguf/
│     ├─ __init__.py
│     ├─ nodes.py
│     └─ requirements.txt
└─ models/
   └─ LLM/
      └─ Qwen-VL/
         ├─ Qwen3.5-9B-Example-Q4_K_M.gguf
         └─ mmproj-Qwen3.5-9B-Example-F16.gguf
```

上面模型名用于说明结构，不是可下载文件的名称。模型可放在 `models/LLM/` 的任意子目录；mmproj 文件名需包含 `mmproj`，才会进入对应下拉框。保留模型名中的系列标识，放好后重启 ComfyUI，或刷新模型列表确认选项更新。

### 如何正确配对

从主模型的下载仓库查找发布者指定的 mmproj。优先使用发布者指定的同一版本配套文件，不要仅凭“都是 Qwen”判断。

- Qwen3.5 **9B** 主模型使用对应 **9B** 的 mmproj。
- Qwen3.8 **27B** 主模型使用对应 **27B** 的 mmproj。
- **不要把 Qwen3.5 9B 主模型配到 Qwen3.8 27B mmproj。**
- 主模型为 `Q8_0`、mmproj 为 `BF16` 可以是正常组合；两者的量化名称不要求相同。
- `FastMTP` / draft 等辅助文件不是普通主模型的替代品；按发布者与后端说明确认具体用途。

只做纯文本且关闭思考时，mmproj 可以为 `None`。本节点开启思考时也需要 mmproj，由视觉聊天处理器控制思考模板。

## 自动下载模型

主节点 `model` 中有四个 `Download | Unsloth Qwen3.8 27B ...` 选项。选择后首次执行会依次下载主模型和 `mmproj-F16.gguf`，自动覆盖本次 mmproj 选择；纯文本执行也会下载配套 mmproj。

来源：[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main)。固定提交为 `4ca720788d1e01f1bff70c033e0d0028fd02e502`，使两份文件来自同一快照。[各预设大小](../README.md#自动下载选项)

```text
ComfyUI/models/LLM/unsloth/Qwen3.8-27B-GGUF/
├─ Qwen3.8-27B-UD-IQ2_S.gguf   （或其他已选量化）
├─ mmproj-F16.gguf
└─ .cache/                    （可能存在的 Hugging Face 下载元数据）
```

下载进度在 ComfyUI 控制台显示。完整文件复用，中断后重新执行可继续；不同预设共用同一份 mmproj。选择一项只下载该量化，不会下载全部四个主模型。

连接 Hugging Face 失败时，也可手动下载相同文件到上述目录。完整、尺寸匹配的文件会复用；使用本地模型时不会为了推理访问 Hugging Face。

模型下载大小不是显存需求，运行还需上下文缓存和视觉处理空间。分片 GGUF 按模型仓库说明完整下载所有分片；四个自动预设均使用单个主模型文件。

## 更新节点

Git 安装的 Windows 用户沿用开头的路径变量：

```powershell
git -C "$ComfyRoot\custom_nodes\ComfyUI-Qwen-gguf" pull
& $ComfyPython -m pip install -r "$ComfyRoot\custom_nodes\ComfyUI-Qwen-gguf\requirements.txt"
```

ZIP 安装的用户重新下载并替换节点代码。模型在 `models/LLM`，与代码分开保存。更新后重启 ComfyUI 并刷新浏览器；旧节点没有新控件时，新建节点并重新连接。是否更换后端 wheel，按更新说明和当前接口决定。
