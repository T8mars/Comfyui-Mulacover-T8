# Validation

Validated on Windows with an NVIDIA GeForce RTX 5090 Laptop GPU (24 GB), CPython 3.12.10, PyTorch 2.10.0+cu128 and NumPy 1.26.4.

- All eight native node classes import without ComfyUI or the YuE2 Studio HTTP service.
- Model loading verifies every indexed MuLaCover and HeartCodec shard, the Qwen embedding weight, the YourMT3 checkpoint and five chord models.
- A real native workflow used `T8MuLaCoverModelLoader`, `T8MuLaCoverMIDICondition`, `T8MuLaCoverStyle` and `T8MuLaCoverGenerate` to produce 4.96 seconds of 48 kHz stereo audio.
- The shared source also completed a 30.0-second audio-conditioned job with melody, chord and drum MIDI outputs in YuE2 Studio v1.5.0.
- Python compilation, example workflow JSON parsing and the no-HTTP-bridge check pass.
