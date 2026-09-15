from __future__ import annotations

from pathlib import Path

from .mulacover_runtime import (
    decode, free_memory, load_pipeline, model_handle, prepare_imports, sample, save_comfy_audio,
    style_tags, temporary_directory, transpose_condition,
)


CATEGORY = "T8/MuLaCover"


class MuLaCoverModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_root": ("STRING", {"default": "", "multiline": False}),
            "device": (["auto", "cuda:0", "cpu"], {"default": "auto"}),
            "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            "lazy_load": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("MULACOVER_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "选择 MuLaCover 权重根目录。留空时使用 ComfyUI/models/MuLaCover-T8。"

    def load(self, model_root, device, dtype, lazy_load):
        return (model_handle(model_root, device, dtype, lazy_load),)


class MuLaCoverStyle:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "topic": ("STRING", {"default": "", "multiline": False}),
            "genre": ("STRING", {"default": "Mandarin pop", "multiline": False}),
            "instrument": ("STRING", {"default": "piano, strings, drums", "multiline": False}),
            "mood": ("STRING", {"default": "warm, hopeful", "multiline": False}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("style_tags",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, topic, genre, instrument, mood):
        return (style_tags(topic, genre, instrument, mood),)


class MuLaCoverAudioCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MULACOVER_MODEL",), "audio": ("AUDIO",),
            "bpm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
            "semitone_shift": ("INT", {"default": 0, "min": -12, "max": 12}),
            "octave_shift": ("INT", {"default": 0, "min": -2, "max": 2}),
        }}
    RETURN_TYPES = ("MULACOVER_CONDITION",)
    RETURN_NAMES = ("condition",)
    FUNCTION = "extract"
    CATEGORY = CATEGORY
    DESCRIPTION = "在 ComfyUI 进程内从歌曲提取旋律、和弦和鼓组条件。"

    def extract(self, model, audio, bpm, semitone_shift, octave_shift):
        source = save_comfy_audio(audio, temporary_directory("source") / "source.wav")
        pipe = load_pipeline(model)
        try:
            inputs = {"ref_audio": str(source)}
            if bpm > 0: inputs["bpm"] = float(bpm)
            condition = pipe._symbolic_condition(inputs)
            return (transpose_condition(condition, int(semitone_shift) + int(octave_shift) * 12),)
        finally:
            pipe = None; free_memory()


class MuLaCoverMIDICondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "melody_midi": ("STRING", {"default": "", "multiline": False}),
            "chord_midi": ("STRING", {"default": "", "multiline": False}),
            "semitone_shift": ("INT", {"default": 0, "min": -12, "max": 12}),
            "octave_shift": ("INT", {"default": 0, "min": -2, "max": 2}),
        }, "optional": {"drum_midi": ("STRING", {"default": "", "multiline": False})}}
    RETURN_TYPES = ("MULACOVER_CONDITION",)
    RETURN_NAMES = ("condition",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, melody_midi, chord_midi, semitone_shift, octave_shift, drum_midi=""):
        prepare_imports()
        from mulacover.symbolic import SymbolicCondition
        condition = SymbolicCondition.from_midi(Path(melody_midi).expanduser(), Path(chord_midi).expanduser(), Path(drum_midi).expanduser() if drum_midi else None)
        return (transpose_condition(condition, int(semitone_shift) + int(octave_shift) * 12),)


class MuLaCoverTransposeCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"condition": ("MULACOVER_CONDITION",),
                             "semitones": ("INT", {"default": 0, "min": -24, "max": 24})}}
    RETURN_TYPES = ("MULACOVER_CONDITION",)
    FUNCTION = "transpose"
    CATEGORY = CATEGORY

    def transpose(self, condition, semitones):
        return (transpose_condition(condition, int(semitones)),)


class MuLaCoverSample:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MULACOVER_MODEL",), "condition": ("MULACOVER_CONDITION",),
            "lyrics": ("STRING", {"default": "[Verse]\n", "multiline": True}),
            "style_tags": ("STRING", {"default": "topic:[unspecified]; genre:[pop]; instrument:[piano, strings, drums]; mood:[warm]", "multiline": True}),
            "duration_seconds": ("INT", {"default": 30, "min": 5, "max": 300}),
            "cfg_scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.05}),
            "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
            "topk": ("INT", {"default": 250, "min": 1, "max": 8191}),
            "seed": ("INT", {"default": 831001, "min": 0, "max": 0xffffffffffffffff}),
        }}
    RETURN_TYPES = ("MULACOVER_TOKENS",)
    RETURN_NAMES = ("music_tokens",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, model, condition, lyrics, style_tags, duration_seconds, cfg_scale, temperature, topk, seed):
        return (sample(model, condition, lyrics, style_tags, duration_seconds, cfg_scale, temperature, topk, seed),)


class MuLaCoverDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"music_tokens": ("MULACOVER_TOKENS",),
                             "decode_seed": ("INT", {"default": 831002, "min": 0, "max": 0xffffffffffffffff})}}
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "output_info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, music_tokens, decode_seed):
        audio, info = decode(music_tokens, decode_seed)
        import json
        return audio, json.dumps(info, ensure_ascii=False, indent=2)


class MuLaCoverGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        values = MuLaCoverSample.INPUT_TYPES()
        values["required"] = dict(values["required"])
        values["required"]["decode_seed"] = ("INT", {"default": 831002, "min": 0, "max": 0xffffffffffffffff})
        return values
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "output_info")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "原生完成重新编曲和解码；生成结果可继续连接 ComfyUI 的音频保存或 RVC 节点。"

    def run(self, model, condition, lyrics, style_tags, duration_seconds, cfg_scale, temperature, topk, seed, decode_seed):
        tokens = sample(model, condition, lyrics, style_tags, duration_seconds, cfg_scale, temperature, topk, seed)
        audio, info = decode(tokens, decode_seed)
        import json
        return audio, json.dumps(info, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "T8MuLaCoverModelLoader": MuLaCoverModelLoader,
    "T8MuLaCoverStyle": MuLaCoverStyle,
    "T8MuLaCoverAudioCondition": MuLaCoverAudioCondition,
    "T8MuLaCoverMIDICondition": MuLaCoverMIDICondition,
    "T8MuLaCoverTransposeCondition": MuLaCoverTransposeCondition,
    "T8MuLaCoverSample": MuLaCoverSample,
    "T8MuLaCoverDecode": MuLaCoverDecode,
    "T8MuLaCoverGenerate": MuLaCoverGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "T8MuLaCoverModelLoader": "MuLaCover 模型加载器 · T8",
    "T8MuLaCoverStyle": "MuLaCover 曲风标签 · T8",
    "T8MuLaCoverAudioCondition": "歌曲提取旋律和弦 · T8",
    "T8MuLaCoverMIDICondition": "MIDI 旋律和弦条件 · T8",
    "T8MuLaCoverTransposeCondition": "旋律移调 · T8",
    "T8MuLaCoverSample": "MuLaCover 生成音乐 Tokens · T8",
    "T8MuLaCoverDecode": "MuLaCover 解码音频 · T8",
    "T8MuLaCoverGenerate": "MuLaCover 一体化重新编曲 · T8",
}
