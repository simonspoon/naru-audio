"""`python -m naru_audio_mlx --socket PATH`: parakeet-mlx, mlx-whisper and
mlx-audio behind the framed protocol (`protocol.py`). For speech-to-text the daemon
does the energy and VAD gating and sends one utterance per `transcribe`,
already at 16 kHz; for text-to-speech `synth` streams mlx-audio's chunks
as they are generated. Each library is imported by the first load that
needs it."""

import argparse
import gc
import os
from contextlib import contextmanager, nullcontext
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

            # IndexTTS needs `tokenizer_name` added to its config, done
            # in a scratch directory rather than on `dir` itself (§ below,
            # `_indextts_load_dir`). Chatterbox's loader also fetches a
            # small shared dependency, mlx-community/S3TokenizerV2,
            # straight from the Hub rather than from `dir`; HF_HUB_OFFLINE
            # would make that raise on a machine that has never cached it.
            # Every other model's weights come entirely from `dir` as-is.
            if header.get("model", "").startswith("indextts"):
                with _indextts_load_dir(Path(header["dir"])) as load_dir:
                    model = load_model(load_dir)
            else:
                with (
                    _network_allowed()
                    if header.get("model", "").startswith("chatterbox")
                    else nullcontext()
                ):
                    model = load_model(Path(header["dir"]))
            answer = {"sample_rate": model.sample_rate}
        elif header.get("model", "").startswith("whisper"):
            from mlx_whisper.load_models import load_model

            # float16, the dtype `mlx_whisper.transcribe` decodes in.
            model = load_model(header["dir"], dtype=mx.float16)
            answer = {}
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
        model = loaded(header)
        if header["model"].startswith("whisper"):
            return transcribe_whisper(model, header, samples)
        from parakeet_mlx.audio import get_logmel

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


# Whisper continues the style of its prompt, so a prompt full of fillers and
# false starts makes it transcribe them instead of cleaning them up.
VERBATIM_PROMPT = "Um, uh, so, I- I mean, like, you know, hmm. Uh-huh."


def transcribe_whisper(model, header, samples):
    """One utterance through mlx-whisper: `header["language"]` when given,
    else Whisper detects it; word timestamps when `header["words"]`.
    `header["verbatim"]` biases it to keep fillers and repetitions: a
    filler-rich `initial_prompt`, and no conditioning on its own previous
    text, which is what lets a hallucinated loop carry on."""
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder

    audio = np.frombuffer(samples or b"", dtype="<f4")
    # `transcribe` takes a path, not a model, and looks it up in a
    # one-slot cache: fill the slot with the loaded model under its name
    # for the call, and empty it after, so an unload frees the weights.
    name = header["model"]
    ModelHolder.model, ModelHolder.model_path = model, name
    options = {}
    if header.get("verbatim"):
        # Without conditioning on previous text, the prompt only biases the
        # first 30 s window of each VAD chunk.
        options = {
            "initial_prompt": VERBATIM_PROMPT,
            "condition_on_previous_text": False,
        }
    try:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=name,
            word_timestamps=bool(header.get("words")),
            language=header.get("language"),
            **options,
        )
    finally:
        ModelHolder.model, ModelHolder.model_path = None, None
    return whisper_answer(result, bool(header.get("words")))


def whisper_answer(result, words):
    """mlx-whisper's `transcribe` result as a `transcribe` answer
    (`protocol.py`): its non-empty segments, each with its words when
    asked for, and the language it decoded in."""
    segments = []
    for s in result["segments"]:
        text = s["text"].strip()
        if not text:
            continue
        segment = {"start": s["start"], "end": s["end"], "text": text}
        if words:
            segment["words"] = [
                {"start": w["start"], "end": w["end"], "text": w["word"].strip()}
                for w in s.get("words", [])
                if w["word"].strip()
            ]
        segments.append(segment)
    answer = {"segments": segments, "language": result.get("language")}
    if words:
        answer["words"] = True
    return answer


def synth(model, header):
    """The chunks of `header["text"]` as f32le bytes, generated one at a
    time as `serve` asks for them."""
    # Chatterbox does not support `speed` at all (mlx-audio's own
    # docstring: "Ignored (Chatterbox doesn't support speed adjustment)");
    # VoxCPM2's, IndexTTS's, OmniVoice's and Breeze's `generate` have no
    # `speed` parameter either, so it would otherwise be swallowed silently
    # by their own `**kwargs` (Breeze's is `**_`, but the effect is the
    # same). Either way, a caller who asks for anything else gets a clear
    # error instead of normal-speed audio it never asked for.
    speed = header.get("speed", 1.0)
    model_name = header.get("model", "")
    if (
        model_name.startswith("chatterbox")
        or model_name.startswith("voxcpm2")
        or model_name.startswith("indextts")
        or model_name.startswith("omnivoice")
        or model_name.startswith("breeze")
    ) and speed != 1.0:
        raise ValueError(
            f"{model_name} does not support speed (got {speed}); "
            "only the default, 1.0, is accepted"
        )
    kwargs = {}
    if header.get("reference"):
        kwargs["ref_audio"] = header["reference"]
    # A cloned voice's transcript: with `ref_audio`, Qwen3-TTS Base clones
    # the voice in context. Chatterbox takes no transcript and ignores it
    # (mlx-audio's `**kwargs`), sent anyway so the request looks the same
    # for every cloning model.
    if header.get("reference_text"):
        kwargs["ref_text"] = header["reference_text"]
    # What the voice should be and how it should speak: VoiceDesign's only
    # voice.
    if header.get("instruct"):
        kwargs["instruct"] = header["instruct"]
    # Chatterbox's emotion-exaggeration dial, 0-1.
    if header.get("exaggeration") is not None:
        kwargs["exaggeration"] = header["exaggeration"]
    # Every other manifest-declared knob (naru task 1458), already checked
    # against the manifest's min/max by the daemon and int-cast there where
    # the kwarg takes one: passed straight through as `generate` kwargs,
    # e.g. Qwen3-TTS's `temperature`/`top_p` or VoxCPM2's `cfg_value`.
    kwargs.update(header.get("knobs") or {})
    # Only Qwen3-TTS's cloning models have the streaming decoder `primed`
    # feeds the reference codes to; Chatterbox has no `speech_tokenizer`
    # and does not stream, so priming would only raise.
    cloned = (
        "ref_audio" in kwargs
        and "ref_text" in kwargs
        and hasattr(model, "speech_tokenizer")
    )
    with primed(model) if cloned else nullcontext():
        for result in model.generate(
            text=header["text"],
            voice=header.get("voice"),
            speed=speed,
            stream=True,
            streaming_interval=STREAMING_INTERVAL,
            **kwargs,
        ):
            yield np.asarray(result.audio, dtype="<f4").tobytes()


@contextmanager
def _indextts_load_dir(model_dir):
    """mlx-community's IndexTTS config.json has no `tokenizer_name` field,
    which mlx-audio's `ModelArgs` requires to find `tokenizer.model`
    (mlx-audio's own test suite fills it in by hand rather than reading it
    from the repo). `model_dir` is a hash-pinned `[[file]]` set (§3.2),
    re-hashed by `naru-audio verify`, so its `config.json` must not be
    touched. Yields a fresh scratch directory instead: every other file
    symlinked in unchanged, plus a `config.json` copy with `tokenizer_name`
    pointing at the scratch directory itself, where the symlinked
    `tokenizer.model` resolves. Removed once the caller is done with it;
    by then `load_model` has already read everything it needs."""
    import json
    import shutil
    import tempfile

    load_dir = Path(tempfile.mkdtemp(prefix="naru-audio-indextts-"))
    try:
        for f in model_dir.iterdir():
            if f.is_file() and f.name != "config.json":
                (load_dir / f.name).symlink_to(f)
        config = json.loads((model_dir / "config.json").read_text())
        config["tokenizer_name"] = str(load_dir)
        (load_dir / "config.json").write_text(json.dumps(config))
        yield load_dir
    finally:
        shutil.rmtree(load_dir, ignore_errors=True)


@contextmanager
def _network_allowed():
    """Lifts `HF_HUB_OFFLINE` for one call, restoring it after: for
    Chatterbox's `snapshot_download` of mlx-community/S3TokenizerV2, so it
    can be fetched and cached under ~/.cache/huggingface the first time,
    like any other `huggingface_hub` download. `huggingface_hub` reads the
    env var once into a module constant at import time, so the env var
    itself is set too (for any subprocess it spawns), but the constant is
    what `is_offline_mode()` actually checks."""
    import huggingface_hub.constants as hf_constants

    prev_env = os.environ.pop("HF_HUB_OFFLINE", None)
    prev_const = hf_constants.HF_HUB_OFFLINE
    hf_constants.HF_HUB_OFFLINE = False
    try:
        yield
    finally:
        hf_constants.HF_HUB_OFFLINE = prev_const
        if prev_env is not None:
            os.environ["HF_HUB_OFFLINE"] = prev_env


@contextmanager
def primed(model):
    """Qwen3-TTS in-context cloning with the reference clip's codes fed
    to the streaming decoder first. Without them the decoder starts from a
    reset state and the first second or so comes out in another voice;
    mlx-audio's non-streaming path prepends them and trims their audio, so
    this does the same: the codes `_prepare_icl_generation_inputs` returns
    go through `streaming_step` once, right after the decoder's first reset
    in `generate`, and their audio is thrown away. Both hooks are instance
    attributes, removed on the way out."""
    # `load_model` wraps the decoder in `mx.compile`, which forwards
    # attribute reads but not writes: hook the module underneath.
    decoder = model.speech_tokenizer.decoder.reset_streaming_state.__self__
    prepare = model._prepare_icl_generation_inputs
    reset = decoder.reset_streaming_state
    reference = []

    def capture(*args, **kwargs):
        inputs = prepare(*args, **kwargs)
        reference.append(inputs[3])
        return inputs

    # `generate` resets again when the stream ends; only the first primes.
    def reset_and_prime():
        reset()
        if reference:
            mx.eval(decoder.streaming_step(reference.pop()))

    model._prepare_icl_generation_inputs = capture
    decoder.reset_streaming_state = reset_and_prime
    try:
        yield
    finally:
        del model._prepare_icl_generation_inputs
        del decoder.reset_streaming_state


def main():
    parser = argparse.ArgumentParser(prog="naru_audio_mlx")
    parser.add_argument("--socket", required=True)
    serve(parser.parse_args().socket, handle)


if __name__ == "__main__":
    main()
