"""`transcribe` for a Whisper model, with `mlx_whisper.transcribe` faked.
Run from mlx/ with the sidecar's venv:
`.venv/bin/python -m unittest tests.test_transcribe_whisper`."""

import unittest
from unittest import mock

import numpy as np

from naru_audio_mlx.__main__ import VERBATIM_PROMPT, transcribe_whisper, whisper_answer

RESULT = {
    "text": " Bonjour Simon.",
    "language": "fr",
    "segments": [
        {
            "start": 0.0,
            "end": 1.5,
            "text": " Bonjour Simon.",
            "words": [
                {"word": " Bonjour", "start": 0.1, "end": 0.6, "probability": 0.9},
                {"word": " Simon.", "start": 0.7, "end": 1.4, "probability": 0.8},
            ],
        },
        {"start": 1.5, "end": 2.0, "text": "  ", "words": []},
    ],
}


class WhisperTest(unittest.TestCase):
    def test_answer_has_words_and_language_when_asked(self):
        answer = whisper_answer(RESULT, True)
        self.assertEqual(answer["language"], "fr")
        self.assertIs(answer["words"], True)
        self.assertEqual(
            answer["segments"],
            [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Bonjour Simon.",
                    "words": [
                        {"start": 0.1, "end": 0.6, "text": "Bonjour", "probability": 0.9},
                        {"start": 0.7, "end": 1.4, "text": "Simon.", "probability": 0.8},
                    ],
                }
            ],
        )

    def test_answer_has_no_words_unless_asked(self):
        answer = whisper_answer(RESULT, False)
        self.assertNotIn("words", answer)
        self.assertNotIn("words", answer["segments"][0])

    def test_segment_whose_words_all_strip_empty_keeps_its_text(self):
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": " ...",
                    "words": [{"word": " ", "start": 0.0, "end": 1.0}],
                }
            ],
        }
        (segment,) = whisper_answer(result, True)["segments"]
        self.assertEqual(segment["text"], "...")
        self.assertEqual(segment["words"], [])

    def test_language_hint_and_words_reach_whisper_and_the_cache_is_emptied(self):
        from mlx_whisper.transcribe import ModelHolder

        model = object()
        samples = np.zeros(1600, dtype="<f4").tobytes()
        seen = {}

        def fake(audio, **kwargs):
            seen.update(kwargs, model=ModelHolder.get_model(kwargs["path_or_hf_repo"], None))
            return RESULT

        with mock.patch("mlx_whisper.transcribe", fake):
            answer = transcribe_whisper(
                model,
                {"model": "whisper-x", "language": "fr", "words": True},
                samples,
            )
        self.assertEqual(answer["language"], "fr")
        self.assertEqual(seen["language"], "fr")
        self.assertIs(seen["word_timestamps"], True)
        self.assertIs(seen["model"], model)
        self.assertIsNone(ModelHolder.model)
        self.assertNotIn("initial_prompt", seen)
        self.assertNotIn("condition_on_previous_text", seen)

    def test_verbatim_passes_a_filler_prompt_and_stops_conditioning(self):
        samples = np.zeros(1600, dtype="<f4").tobytes()
        seen = {}

        def fake(audio, **kwargs):
            seen.update(kwargs)
            return RESULT

        with mock.patch("mlx_whisper.transcribe", fake):
            transcribe_whisper(
                object(),
                {"model": "whisper-x", "words": True, "verbatim": True},
                samples,
            )
        self.assertEqual(seen["initial_prompt"], VERBATIM_PROMPT)
        self.assertIn("um", seen["initial_prompt"].lower())
        self.assertIs(seen["condition_on_previous_text"], False)
        self.assertIs(seen["word_timestamps"], True)
        # Disfluencies are never suppressed: no suppression override at all.
        self.assertNotIn("suppress_tokens", seen)
        self.assertNotIn("suppress_blank", seen)


if __name__ == "__main__":
    unittest.main()
