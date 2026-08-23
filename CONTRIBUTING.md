# Contributing

Issues and pull requests are welcome.

## Dev setup (Windows)

```bat
py -3 -m pip install -r requirements.txt pytest
py -3 -m pytest tests -q
py -3 -m fluid_motion
```

## Notes

- Keep changes scoped. This is a tray manager that injects a VapourSynth filter into mpv; it is not a player.
- UI copy is Traditional Chinese. Match that unless you are adding a locale layer.
- Do not commit `dist/`, `build/`, `*.exe`, `*.onnx`, or TensorRT engines.
- Realtime interpolation requires an NVIDIA GPU plus vs-mlrt TensorRT. Tests do not require a GPU.
