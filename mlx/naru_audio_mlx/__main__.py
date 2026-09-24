"""`python -m naru_audio_mlx --socket PATH`: parakeet-mlx behind the
framed protocol (`protocol.py`). The daemon does the energy and VAD gating
and sends one utterance per `transcribe`, already at 16 kHz."""

import argparse
import gc

import mlx.core as mx
import numpy as np
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel

from .protocol import serve

models = {}


def handle(header, samples):
    op = header.get("op")
    if op == "load":
        # A local directory with config.json and model.safetensors; the
        # daemon sets HF_HUB_OFFLINE, so nothing is downloaded.
        model = from_pretrained(header["dir"])
        # MLX loads lazily: materialise the weights now, so the load pays
        # for them and `stats` counts them.
        mx.eval(model.parameters())
        models[header["model"]] = model
        return {}
    if op == "unload":
        models.pop(header["model"], None)
        gc.collect()
        mx.clear_cache()
        return {}
    if op == "transcribe":
        model = models.get(header["model"])
        if model is None:
            raise KeyError(f"{header['model']} is not loaded")
        audio = mx.array(np.frombuffer(samples or b"", dtype="<f4"))
        # Shorter than one hop, there is no frame to decode.
        if audio.size < model.preprocessor_config.hop_length:
            return {"segments": []}
        result = model.generate(get_logmel(audio, model.preprocessor_config))[0]
        segments = [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in result.sentences
            if s.text.strip()
        ]
        return {"segments": segments}
    if op == "stats":
        return {"active_bytes": mx.get_active_memory()}
    raise ValueError(f"unknown op {op!r}")


def main():
    parser = argparse.ArgumentParser(prog="naru_audio_mlx")
    parser.add_argument("--socket", required=True)
    serve(parser.parse_args().socket, handle)


main()
