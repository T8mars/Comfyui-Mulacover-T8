# ComfyUI-Mulacover-T8

MuLaCover 的原生 ComfyUI 节点。它在 ComfyUI 的 Python 进程内直接加载模型，不依赖网页服务，也不通过 HTTP 桥接。

## 来源与致谢

**主要感谢 MuLaCover 原作者 [HeartMuLa 团队及贡献者](https://github.com/HeartMuLa/MuLaCover)** 开源 MuLaCover 模型、推理代码和生成指南。本项目是围绕上游 MuLaCover 的 ComfyUI 原生节点适配层；论文、模型说明与上游更新请以上游仓库为准。

- 上游 MuLaCover 代码：[HeartMuLa/MuLaCover](https://github.com/HeartMuLa/MuLaCover)
- MuLaCover 权重：[HeartMuLa/MuLaCover](https://huggingface.co/HeartMuLa/MuLaCover)
- HeartCodec 权重：[HeartMuLa/HeartCodec-oss-20260123](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123)
- 符号转录依赖：[YourMT3](https://github.com/magenta/mt3) 及 MuLaCover 使用的 ChordNet 检查点

本仓库的 `vendor/mulacover` 保留上游 Apache-2.0 许可文件；模型权重及生成结果遵循上游 `MODEL_LICENSE` 的非商业使用条款。`nodes.py`、`mulacover_runtime.py`、模型路径复用、ComfyUI 数据类型和工作流示例是 T8star 为 ComfyUI 编写的适配代码，未声称拥有上游模型或算法的版权。感谢 HeartMuLa 让本项目能够在本地音乐工作台和 ComfyUI 中复用这些能力。

## 能做什么

- 从参考歌曲提取旋律、和弦和鼓组，再用新歌词和新曲风生成完整歌曲。
- 直接读取旋律 MIDI、和弦 MIDI 和可选鼓组 MIDI。
- 在生成前按半音或八度调整旋律，适配男女声音域。
- 将采样和解码拆开，固定生成种子与解码种子进行可复现对比。
- 输出标准 ComfyUI `AUDIO`，可继续连接保存音频、RVC、Seed-VC 或其他音频节点。

## 安装

把本目录放入 `ComfyUI/custom_nodes/Comfyui-Mulacover-T8`，然后在 ComfyUI 的 Python 环境中运行：

```powershell
python -m pip install -r requirements.txt
```

现有 PyTorch 与 CUDA 应由 ComfyUI 管理，不要为本节点重复安装另一套 PyTorch。

## 模型目录

默认位置为 `ComfyUI/models/MuLaCover-T8`。也可以在“MuLaCover 模型加载器 · T8”中填写 YuE2 Studio 完整整合包的 `models` 绝对路径，共用同一份权重。

```text
MuLaCover-T8/
├── MuLaCover/
├── HeartCodec-oss/
├── Qwen3-Embedding-0.6B/
└── SymbolicTranscriptor/
    ├── yourmt3/last.ckpt
    └── chord/*.best.sdict
```

安装 `huggingface_hub` 后可自动下载：

```powershell
python download_models.py --model-root "D:\ComfyUI\models\MuLaCover-T8"
```

参考音频路径需要 `SymbolicTranscriptor`；只用 MIDI 时不会加载转谱组件。

## 推荐工作流

参考歌曲重新编曲：

1. `Load Audio` → `歌曲提取旋律和弦 · T8`
2. `MuLaCover 曲风标签 · T8`
3. 上述两项与模型、歌词连接到 `MuLaCover 一体化重新编曲 · T8`
4. 输出 `AUDIO` 连接 ComfyUI 音频保存节点，或继续连接 RVC/Seed-VC

高级工作流可使用 `MuLaCover 生成音乐 Tokens · T8` → `MuLaCover 解码音频 · T8`，只更换解码种子时可以复用已经生成的音乐 tokens。

仓库中的 [`examples/mulacover-audio-remix.json`](examples/mulacover-audio-remix.json) 提供了可导入的基础节点布局；导入后把 ComfyUI 的 `Load Audio` 连接到音频条件节点即可。

参数建议：先用 30 秒、`CFG 1.5`、`Temperature 1.0`、`Top-K 250`。男声演唱过高时把旋律降低 12 半音；女声需要更高音域时升高 12 半音。

已在 Windows、RTX 5090 Laptop 24GB、Python 3.12.10、Torch 2.10.0+cu128 环境用上述原生节点链实测生成 48 kHz 双声道音频。参考音频转谱和 30 秒完整工作台任务也已通过；详细记录见主项目的 [VALIDATION.md](https://github.com/T8mars/Comfyui-YuE2-T8/blob/main/VALIDATION.md)。

## Attribution and thanks

**Primary thanks go to the original MuLaCover authors, the [HeartMuLa team and contributors](https://github.com/HeartMuLa/MuLaCover)**, for releasing the MuLaCover model, inference implementation and documentation. This plugin is a native ComfyUI integration layer around the upstream project; its repository remains the authoritative source for model updates, research details and licensing.

The vendored `vendor/mulacover` code keeps its Apache-2.0 license. Model weights and generated audio follow the upstream `MODEL_LICENSE` and are restricted to non-commercial use. The T8star-authored files provide ComfyUI node definitions, local model-path reuse, condition wiring and workflow examples; they do not claim ownership of the upstream model or algorithm.

## T8 项目链接

- 主项目与源码：[Comfyui-YuE2-T8](https://github.com/T8mars/Comfyui-YuE2-T8)
- B站：[T8star-Aix](https://space.bilibili.com/385085361)
- YouTube：[T8star-Aix](https://www.youtube.com/@T8star-Aix/)
- API 注册：[Seedance API](https://api.seedance.nz/sign-up?aff=5f4w)
- 在线 AI 应用：[RunningHub](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- ComfyUI 整合包：[夸克网盘](https://pan.quark.cn/s/264edb7e36bd)
- Hugging Face：[T8star](https://huggingface.co/t8star)
- YuE2 模型仓库：[YuE2-Comfy](https://huggingface.co/t8star/YuE2-Comfy)