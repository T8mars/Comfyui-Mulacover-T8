from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_WEIGHTS = {
    "MuLaCover/model-00001-of-00005.safetensors": 1_994_713_352,
    "MuLaCover/model-00002-of-00005.safetensors": 1_981_941_248,
    "MuLaCover/model-00003-of-00005.safetensors": 1_963_054_632,
    "MuLaCover/model-00004-of-00005.safetensors": 1_999_477_480,
    "MuLaCover/model-00005-of-00005.safetensors": 809_619_536,
    "HeartCodec-oss/model-00001-of-00002.safetensors": 4_930_472_472,
    "HeartCodec-oss/model-00002-of-00002.safetensors": 1_707_911_436,
    "Qwen3-Embedding-0.6B/model.safetensors": 1_191_586_416,
    "SymbolicTranscriptor/yourmt3/last.ckpt": 561_544_628,
}


def prepare_imports() -> None:
    vendor = ROOT / "vendor" / "mulacover"
    for directory in (vendor / "compat", vendor / "src"):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)


def default_model_root() -> Path:
    try:
        import folder_paths
        return Path(folder_paths.models_dir) / "MuLaCover-T8"
    except ImportError:
        return ROOT / "models"


def model_handle(model_root: str, device: str, dtype: str, lazy_load: bool) -> dict:
    import torch

    path = Path(os.path.expandvars(model_root.strip())).expanduser() if model_root.strip() else default_model_root()
    path = path.resolve()
    required = (
        "MuLaCover/config.json", "MuLaCover/gen_config.json",
        "MuLaCover/model.safetensors.index.json", "MuLaCover/tokenizer.json",
        "HeartCodec-oss/config.json", "HeartCodec-oss/model.safetensors.index.json",
        "Qwen3-Embedding-0.6B/config.json", "Qwen3-Embedding-0.6B/model.safetensors",
        "SymbolicTranscriptor/yourmt3/last.ckpt",
    )
    missing = [name for name in required if not (path / name).is_file() or not (path / name).stat().st_size]
    for index_name in ("MuLaCover/model.safetensors.index.json", "HeartCodec-oss/model.safetensors.index.json"):
        index_path = path / index_name
        if index_path.is_file():
            try:
                weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
                directory = index_path.parent
                for filename in sorted(set(weight_map.values())):
                    candidate = directory / filename
                    relative = candidate.relative_to(path).as_posix()
                    if not candidate.is_file() or not candidate.stat().st_size:
                        missing.append(relative)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                missing.append(index_name + "（索引无效）")
    chord_dir = path / "SymbolicTranscriptor" / "chord"
    if len(list(chord_dir.glob("*.best.sdict"))) < 5:
        missing.append("SymbolicTranscriptor/chord/*.best.sdict（需要 5 个）")
    for name, expected in EXPECTED_WEIGHTS.items():
        candidate = path / name
        if not candidate.is_file() or candidate.stat().st_size != expected:
            missing.append(f"{name}（应为 {expected} 字节）")
    if missing:
        raise FileNotFoundError(f"MuLaCover 模型目录不完整：{', '.join(missing)}；当前目录：{path}")
    resolved_device = "cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用的 NVIDIA CUDA 显卡")
    return {"model_root": str(path), "device": resolved_device, "dtype": dtype, "lazy_load": bool(lazy_load)}


def load_pipeline(handle: dict):
    prepare_imports()
    import torch
    from mulacover import MuLaCoverGenPipeline

    device = torch.device(handle["device"])
    main_dtype = torch.float32 if device.type == "cpu" else getattr(torch, handle["dtype"])
    return MuLaCoverGenPipeline.from_pretrained(
        handle["model_root"], device=device,
        dtype={"mulacover": main_dtype, "codec": torch.float32,
               "qwen": torch.float32, "transcriptor": torch.float32},
        lazy_load=handle.get("lazy_load", True),
    )


def clean_style(value: str) -> str:
    return " ".join(str(value or "").replace("[", "(").replace("]", ")").split())


def style_tags(topic: str, genre: str, instrument: str, mood: str) -> str:
    values = {"topic": clean_style(topic), "genre": clean_style(genre),
              "instrument": clean_style(instrument), "mood": clean_style(mood)}
    if not any(values.values()):
        raise ValueError("流派、乐器、情绪和主题至少填写一项")
    return "; ".join(f"{key}:[{value or 'unspecified'}]" for key, value in values.items())


def transpose_condition(condition, semitones: int):
    if not semitones:
        return condition
    prepare_imports()
    from mulacover.symbolic import SymbolicCondition

    melody = condition.melody.clone()
    melody[:, 1] = (melody[:, 1] + semitones).clamp(0, 127)
    chords = condition.chords.clone()
    if len(chords):
        chords[:, 1] = (chords[:, 1] + semitones) % 12
    return SymbolicCondition(melody=melody, chords=chords,
                             drums=condition.drums.clone(), bpm=condition.bpm)


def temporary_directory(prefix: str) -> Path:
    try:
        import folder_paths
        base = Path(folder_paths.get_temp_directory())
    except ImportError:
        base = Path(tempfile.gettempdir())
    result = base / f"mulacover-{prefix}-{uuid.uuid4().hex[:12]}"
    result.mkdir(parents=True, exist_ok=True)
    return result


def save_comfy_audio(audio: dict, destination: Path) -> Path:
    import soundfile as sf
    import torch

    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if not isinstance(waveform, torch.Tensor) or sample_rate <= 0:
        raise ValueError("AUDIO 输入无效")
    value = waveform.detach().float().cpu()
    if value.ndim == 3:
        value = value[0]
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError("AUDIO 波形维度无效")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, value.numpy().T, sample_rate)
    return destination


def comfy_audio(waveform, sample_rate: int) -> dict:
    value = waveform.detach().float().cpu()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    return {"waveform": value, "sample_rate": int(sample_rate)}


def interrupt() -> None:
    try:
        import comfy.model_management as model_management
        model_management.throw_exception_if_processing_interrupted()
    except (ImportError, AttributeError):
        return


def free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def progress_bar(total: int):
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(max(1, int(total)))
    except ImportError:
        class Empty:
            def update_absolute(self, *_args, **_kwargs):
                return None
        return Empty()


def sample(handle: dict, condition, lyrics: str, tags: str, duration_seconds: int,
           cfg_scale: float, temperature: float, topk: int, seed: int) -> dict:
    import torch

    if not str(lyrics).strip():
        raise ValueError("歌词不能为空")
    interrupt()
    try:
        import comfy.model_management as model_management
        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except ImportError:
        pass
    pipe = load_pipeline(handle)
    directory = temporary_directory("condition")
    paths = condition.save_midi(directory)
    model_inputs = pipe.preprocess(
        {"lyrics": lyrics, "tags": tags, "melody_midi": str(paths["melody"]),
         "chord_midi": str(paths["chord"]), "drum_midi": str(paths["drums"])},
        cfg_scale=float(cfg_scale), max_audio_length_ms=int(duration_seconds) * 1000,
    )
    bar = progress_bar(int(duration_seconds) * 1000 // 80)
    def on_progress(_stage, completed, total):
        interrupt(); bar.update_absolute(completed, total)
    device = torch.device(handle["device"])
    index = device.index if device.type == "cuda" and device.index is not None else (torch.cuda.current_device() if device.type == "cuda" else None)
    devices = [index] if index is not None else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            outputs = pipe._forward(
                model_inputs, max_audio_length_ms=int(duration_seconds) * 1000,
                temperature=float(temperature), topk=int(topk), cfg_scale=float(cfg_scale),
                disable_progress=True, cancelled=lambda: False, on_progress=on_progress,
            )
        return {"frames": outputs["frames"], "model": dict(handle), "condition_paths": {key: str(value) for key, value in paths.items()},
                "seed": int(seed), "duration_seconds": int(duration_seconds)}
    finally:
        pipe = None; free_memory()


def decode(tokens: dict, decode_seed: int) -> tuple[dict, dict]:
    import torch

    handle = tokens["model"]
    pipe = load_pipeline(handle)
    directory = temporary_directory("output")
    output = directory / "remix.flac"
    bar = progress_bar(10)
    def on_progress(_stage, completed, total):
        interrupt(); bar.update_absolute(completed, total)
    try:
        decoded = pipe.postprocess(
            {"frames": tokens["frames"]}, save_path=output, disable_progress=True,
            cancelled=lambda: False, on_progress=on_progress, decode_seed=int(decode_seed),
        )
        return comfy_audio(decoded["waveform"], decoded["sample_rate"]), {
            "audio_path": str(output), **tokens.get("condition_paths", {}),
            "seed": tokens.get("seed"), "decode_seed": int(decode_seed),
        }
    finally:
        pipe = None; free_memory()
