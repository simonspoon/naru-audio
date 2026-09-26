"""`synth` against a fake Qwen3-TTS that records what its streaming
decoder is fed. Run from mlx/ with the sidecar's venv:
`.venv/bin/python -m unittest tests.test_synth`."""

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from naru_audio_mlx.__main__ import _indextts_load_dir, synth

REFERENCE = mx.full((1, 16, 3), 7)
GENERATED = mx.full((1, 16, 2), 1)


class Decoder(nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.record = calls.append

    def __call__(self, codes):
        return codes

    def reset_streaming_state(self):
        self.record("reset")

    # One sample per code, valued by the code.
    def streaming_step(self, codes):
        self.record(codes.shape[2])
        return codes[:, :1, :].astype(mx.float32)


class Tokenizer:
    def __init__(self, calls):
        self.decoder = mx.compile(Decoder(calls))


class Model:
    """mlx-audio's Qwen3-TTS in outline: a compiled decoder, in-context
    cloning when given a reference clip and its text, the decoder reset
    before and after the stream."""

    def __init__(self):
        self.calls = []
        self.speech_tokenizer = Tokenizer(self.calls)

    def _prepare_icl_generation_inputs(self, **kwargs):
        return None, None, None, REFERENCE

    def generate(self, text, ref_audio=None, ref_text=None, **kwargs):
        if ref_audio is not None and ref_text is not None:
            self._prepare_icl_generation_inputs(ref_audio=ref_audio, ref_text=ref_text)
        decoder = self.speech_tokenizer.decoder
        decoder.reset_streaming_state()
        wav = decoder.streaming_step(GENERATED)
        yield type("Result", (), {"audio": wav.squeeze(1)[0]})
        decoder.reset_streaming_state()


def audio(model, header):
    chunks = b"".join(synth(model, {"text": "Hi.", **header}))
    return np.frombuffer(chunks, dtype="<f4").tolist()


class Synth(unittest.TestCase):
    def test_cloned_voice_primes_the_decoder_with_the_reference_once(self):
        model = Model()
        header = {"reference": "ref.wav", "reference_text": "Hello there."}
        self.assertEqual(audio(model, header), [1.0, 1.0])
        # The reference's 3 codes right after the first reset, then only
        # the generated 2; the closing reset primes nothing.
        self.assertEqual(model.calls, ["reset", 3, 2, "reset"])
        # The hooks are gone once the stream ends.
        module = model.speech_tokenizer.decoder.reset_streaming_state.__self__
        self.assertNotIn("reset_streaming_state", vars(module))
        self.assertNotIn("_prepare_icl_generation_inputs", vars(model))

    def test_other_voices_are_not_primed(self):
        for header in [{}, {"voice": "ryan"}, {"reference": "ref.wav"}]:
            model = Model()
            self.assertEqual(audio(model, header), [1.0, 1.0])
            self.assertEqual(model.calls, ["reset", 2, "reset"], header)


class Chatterbox:
    """mlx-audio's Chatterbox in outline: no `speech_tokenizer`, ignores
    `voice`, takes `exaggeration`, yields once, non-streaming."""

    def generate(self, text, ref_audio=None, exaggeration=None, **kwargs):
        self.seen_exaggeration = exaggeration
        yield type("Result", (), {"audio": mx.full((1,), 2.0)})


class ChatterboxSynth(unittest.TestCase):
    def test_reference_and_transcript_do_not_prime_a_model_with_no_speech_tokenizer(
        self,
    ):
        model = Chatterbox()
        header = {
            "model": "chatterbox-tts-8bit-mlx",
            "reference": "ref.wav",
            "reference_text": "Hello there.",
        }
        # No AttributeError from `primed` reaching for `speech_tokenizer`.
        self.assertEqual(audio(model, header), [2.0])

    def test_exaggeration_is_passed_through(self):
        model = Chatterbox()
        header = {"model": "chatterbox-tts-8bit-mlx", "exaggeration": 0.8}
        audio(model, header)
        self.assertEqual(model.seen_exaggeration, 0.8)

    def test_non_default_speed_raises(self):
        model = Chatterbox()
        header = {"model": "chatterbox-tts-8bit-mlx", "speed": 1.5}
        with self.assertRaises(ValueError):
            audio(model, header)

    def test_default_speed_is_accepted(self):
        model = Chatterbox()
        header = {"model": "chatterbox-tts-8bit-mlx", "speed": 1.0}
        self.assertEqual(audio(model, header), [2.0])


class VoxCPM2:
    """mlx-audio's VoxCPM2 in outline: no `speech_tokenizer`, clones with
    `ref_audio` and designs with `instruct`, has no `speed` parameter at
    all (swallowed by `**kwargs` if sent), yields once, non-streaming."""

    def generate(self, text, ref_audio=None, instruct=None, **kwargs):
        self.seen_ref_audio = ref_audio
        self.seen_instruct = instruct
        yield type("Result", (), {"audio": mx.full((1,), 3.0)})


class VoxCPM2Synth(unittest.TestCase):
    def test_reference_clones_without_priming(self):
        model = VoxCPM2()
        header = {
            "model": "voxcpm2-8bit-mlx",
            "reference": "ref.wav",
            "reference_text": "Hello there.",
        }
        self.assertEqual(audio(model, header), [3.0])
        self.assertEqual(model.seen_ref_audio, "ref.wav")

    def test_instruct_is_passed_through_for_voice_design(self):
        model = VoxCPM2()
        header = {"model": "voxcpm2-8bit-mlx", "instruct": "A calm narrator."}
        audio(model, header)
        self.assertEqual(model.seen_instruct, "A calm narrator.")

    def test_non_default_speed_raises(self):
        model = VoxCPM2()
        header = {"model": "voxcpm2-8bit-mlx", "speed": 1.5}
        with self.assertRaises(ValueError):
            audio(model, header)

    def test_default_speed_is_accepted(self):
        model = VoxCPM2()
        header = {"model": "voxcpm2-8bit-mlx", "speed": 1.0}
        self.assertEqual(audio(model, header), [3.0])


class IndexTTS:
    """mlx-audio's IndexTTS in outline: no `speech_tokenizer`, clones with
    `ref_audio`, has no `speed` or `ref_text` parameter at all (swallowed
    by `**kwargs` if sent), yields once, non-streaming."""

    def generate(self, text, ref_audio=None, **kwargs):
        self.seen_ref_audio = ref_audio
        yield type("Result", (), {"audio": mx.full((1,), 4.0)})


class IndexTTSSynth(unittest.TestCase):
    def test_reference_clones_without_priming(self):
        model = IndexTTS()
        header = {
            "model": "indextts-1.5-mlx",
            "reference": "ref.wav",
            "reference_text": "Hello there.",
        }
        self.assertEqual(audio(model, header), [4.0])
        self.assertEqual(model.seen_ref_audio, "ref.wav")

    def test_non_default_speed_raises(self):
        model = IndexTTS()
        header = {"model": "indextts-1.5-mlx", "speed": 1.5}
        with self.assertRaises(ValueError):
            audio(model, header)

    def test_default_speed_is_accepted(self):
        model = IndexTTS()
        header = {"model": "indextts-1.5-mlx", "speed": 1.0}
        self.assertEqual(audio(model, header), [4.0])


class OmniVoice:
    """mlx-audio's OmniVoice in outline: no `speech_tokenizer`, clones with
    `ref_audio` and designs with `instruct`, has no `speed` parameter at
    all (swallowed by `**kwargs` if sent), yields once, non-streaming."""

    def generate(self, text, ref_audio=None, instruct=None, **kwargs):
        self.seen_ref_audio = ref_audio
        self.seen_instruct = instruct
        yield type("Result", (), {"audio": mx.full((1,), 5.0)})


class OmniVoiceSynth(unittest.TestCase):
    def test_reference_clones_without_priming(self):
        model = OmniVoice()
        header = {
            "model": "omnivoice-bf16-mlx",
            "reference": "ref.wav",
            "reference_text": "Hello there.",
        }
        self.assertEqual(audio(model, header), [5.0])
        self.assertEqual(model.seen_ref_audio, "ref.wav")

    def test_instruct_is_passed_through_for_voice_design(self):
        model = OmniVoice()
        header = {"model": "omnivoice-bf16-mlx", "instruct": "A calm narrator."}
        audio(model, header)
        self.assertEqual(model.seen_instruct, "A calm narrator.")

    def test_non_default_speed_raises(self):
        model = OmniVoice()
        header = {"model": "omnivoice-bf16-mlx", "speed": 1.5}
        with self.assertRaises(ValueError):
            audio(model, header)

    def test_default_speed_is_accepted(self):
        model = OmniVoice()
        header = {"model": "omnivoice-bf16-mlx", "speed": 1.0}
        self.assertEqual(audio(model, header), [5.0])


class IndexTTSLoadDir(unittest.TestCase):
    """`_indextts_load_dir` must never touch the pulled model directory:
    its `config.json` is one of the manifest's hash-pinned `[[file]]`s,
    re-hashed by `naru-audio verify`."""

    def setUp(self):
        self.model_dir = Path(tempfile.mkdtemp())
        self.config = {"gpt": {}, "bigvgan": {}}
        (self.model_dir / "config.json").write_text(json.dumps(self.config))
        (self.model_dir / "tokenizer.model").write_bytes(b"fake tokenizer")
        (self.model_dir / "model.safetensors").write_bytes(b"fake weights")

    def test_model_dir_config_is_untouched(self):
        original = (self.model_dir / "config.json").read_bytes()
        with _indextts_load_dir(self.model_dir):
            pass
        self.assertEqual((self.model_dir / "config.json").read_bytes(), original)
        self.assertEqual(
            json.loads(original), self.config, "config.json gained a field"
        )

    def test_scratch_dir_has_tokenizer_name_and_symlinked_files(self):
        with _indextts_load_dir(self.model_dir) as load_dir:
            config = json.loads((load_dir / "config.json").read_text())
            self.assertEqual(config["tokenizer_name"], str(load_dir))
            self.assertEqual(
                (load_dir / "tokenizer.model").read_bytes(), b"fake tokenizer"
            )
            self.assertTrue((load_dir / "tokenizer.model").is_symlink())
        # Removed once the caller is done with it.
        self.assertFalse(load_dir.exists())

    def test_two_loads_each_get_their_own_scratch_dir(self):
        with _indextts_load_dir(self.model_dir) as first:
            first_path = first
        with _indextts_load_dir(self.model_dir) as second:
            self.assertNotEqual(first_path, second)
            self.assertTrue(second.exists())
        self.assertFalse(first_path.exists())
        self.assertFalse(second.exists())


if __name__ == "__main__":
    unittest.main()
