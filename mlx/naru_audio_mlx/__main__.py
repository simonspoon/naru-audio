"""`python -m naru_audio_mlx --socket PATH`: parakeet-mlx and mlx-audio
behind the framed protocol (`protocol.py`). For speech-to-text the daemon
does the energy and VAD gating and sends one utterance per `transcribe`,
already at 16 kHz; for text-to-speech `synth` streams mlx-audio's chunks
as they are generated. Each library is imported by the first load that
needs it."""

import argparse
import gc
from pathlib import Path

import mlx.core as mx
import numpy as np

from .protocol import serve

# The seconds of audio in each `synth` chunk: mlx-audio's streaming
# interval, so the first audio is out after this much is generated.
STREAMING_INTERVAL = 0.32

models = {}


def handle(header, samples):
    op = header.get("op")
    if op == "load":
        # A local directory with the model's files; the daemon sets
        # HF_HUB_OFFLINE, so nothing is downloaded.
        if header.get("kind") == "tts":
            from mlx_audio.tts.utils import load_model

            model = load_model(Path(header["dir"]))
            answer = {"sample_rate": model.sample_rate}
        else:
            from parakeet_mlx import from_pretrained

            model = from_pretrained(header["dir"])
            answer = {}
        # MLX loads lazily: materialise the weights now, so the load pays
        # for them and `stats` counts them.
        mx.eval(model.parameters())
        models[header["model"]] = model
        return answer
    if op == "unload":
        models.pop(header["model"], None)
        gc.collect()
        mx.clear_cache()
        return {}
    if op == "transcribe":
        from parakeet_mlx.audio import get_logmel

        model = loaded(header)
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
    if op == "synth":
        return synth(loaded(header), header)
    if op == "stats":
        return {"active_bytes": mx.get_active_memory()}
    raise ValueError(f"unknown op {op!r}")


def loaded(header):
    model = models.get(header["model"])
    if model is None:
        raise KeyError(f"{header['model']} is not loaded")
    return model


def synth(model, header):
    """The chunks of `header["text"]` as f32le bytes, generated one at a
    time as `serve` asks for them."""
    kwargs = {}
    if header.get("reference"):
        kwargs["ref_audio"] = header["reference"]
    # A cloned voice's transcript: with `ref_audio`, Qwen3-TTS Base clones
    # the voice in context.
    if header.get("reference_text"):
        kwargs["ref_text"] = header["reference_text"]
    # What the voice should be and how it should speak: VoiceDesign's only
    # voice.
    if header.get("instruct"):
        kwargs["instruct"] = header["instruct"]
    for result in model.generate(
        text=header["text"],
        voice=header.get("voice"),
        speed=header.get("speed", 1.0),
        stream=True,
        streaming_interval=STREAMING_INTERVAL,
        **kwargs,
    ):
        yield np.asarray(result.audio, dtype="<f4").tobytes()


def main():
    parser = argparse.ArgumentParser(prog="naru_audio_mlx")
    parser.add_argument("--socket", required=True)
    serve(parser.parse_args().socket, handle)


main()
