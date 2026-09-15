# ComfyUI-Mulacover-T8

MuLaCover 的原生 ComfyUI 节点。它在 ComfyUI 的 Python 进程内直接加载模型，不依赖网页服务，也不通过 HTTP 桥接。

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
