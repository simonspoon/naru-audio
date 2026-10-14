# Voice design prompting

How to write the `instructions` string for each design-capable model. The admin
Design tab (`src/admin/js/design-models.mjs`) encodes the same rules as a
per-model form with starter presets. Results vary run to run on every model:
compare several takes.

## Qwen3-TTS VoiceDesign (`qwen3-tts-1.7b-voicedesign-mlx`)

- `instruct` is free-form natural language, separate from the spoken text.
- Length: the hosted API limits it to 500 characters. That is soft guidance for
  the open model; the Design tab warns above 500.
- Combine several dimensions with concrete words:
  - gender: male, female, neutral
  - age: child, teenager, young adult, middle-aged, senior
  - pitch: high / medium / low, "slightly high", "slightly low"
  - speed: fast / medium / slow, "slightly fast", "slightly slow"
  - emotion: cheerful, calm, gentle, serious, lively, composed, soothing
  - timbre: resonant, crisp, husky, mellow, sweet, deep, powerful
  - scene: news broadcast, advertising, audiobook, animation character, voice
    assistant, documentary narration
- Avoid vague words ("nice", "normal"), "female voice" alone, and celebrity
  imitation.
- Accents are weak. Chinese and English are the benchmarked languages. Keep the
  voice description out of the spoken text.
- Examples:
  - `Female, young adult, slightly high pitch, fast pace, cheerful and lively, like a fashion product presenter`
  - `Male, middle-aged, low and resonant, slow and calm, documentary narrator`
  - `Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.`

## VoxCPM2 (`voxcpm2-8bit-mlx`)

- The model reads the description as a parenthesised prefix of the text:
  `(A young woman, gentle and sweet voice)Hello...`. mlx-audio's `voxcpm2.py`
  `generate` adds the parentheses itself from `instruct`, so send the bare
  description as `instructions`. Wrapping it yourself doubles them.
- Open vocabulary: age, gender, pitch, pace, emotional tone, vocal texture, in
  Chinese or English. 30 languages plus 9 Chinese dialects. For a dialect, name
  it in the description (for example 广东话) and write the text in dialect wording.
  The dialect list in the Design tab is a set of common examples, not a verified
  list of the nine.
- Very short texts (under about 1 second of speech) sound weak; long text
  drifts, so split it. Try 1 to 3 takes.
- Examples: `A warm, gentle female voice in her 30s with a calm and soothing tone`,
  `an excited young man`.

## OmniVoice (`omnivoice-bf16-mlx`)

Licence: CC-BY-NC, non-commercial only.

- Closed vocabulary. Comma-separated, case-insensitive, at most one value per
  category. Half-width `,` for English, full-width `，` for Chinese. Partial
  strings such as `female` are fine.
  - gender: male, female
  - age: child, teenager, young adult, middle-aged, elderly
  - pitch: very low pitch, low pitch, moderate pitch, high pitch, very high pitch
  - style: whisper
  - English accent (English text only): american, british, australian, canadian,
    indian, chinese, korean, japanese, portuguese, russian, each written
    `<x> accent`
  - Chinese dialect (Chinese text only): 河南话, 陕西话, 四川话, 贵州话, 云南话,
    桂林话, 济南话, 石家庄话, 甘肃话, 宁夏话, 青岛话, 东北话
- Accent has no effect on Chinese text; dialect has none on English text. Voice
  design was trained on Chinese and English only.
- Inline tags go in the spoken text, not the description: `[laughter]` `[sigh]`
  `[confirmation-en]` `[question-en]` `[question-ah]` `[question-oh]`
  `[question-ei]` `[question-yi]` `[surprise-ah]` `[surprise-oh]` `[surprise-wa]`
  `[surprise-yo]` `[dissatisfaction-hnn]`.
- Examples: `female, young adult, high pitch, british accent`;
  `male, elderly, low pitch, whisper`.

## Breeze-TTS-2 (`breeze-tts-2-mlx`)

By BreezeBlue (not MediaTek). Research / non-commercial licence.

- Free-text natural language. Match the instruction language to the text
  language (English or Chinese).
- Tags in the spoken text: English `(laugh)` `(cough)` `(clears throat)`
  `(sigh)`; Chinese `[笑]` `[咳嗽]` `[清嗓子]` `[叹气]`.
- Audio is capped at about 120 seconds.
- Examples: `A warm, thoughtful young woman with a clear voice and a calm, reflective delivery.`;
  `一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。`
