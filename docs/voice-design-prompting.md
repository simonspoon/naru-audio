# Voice design prompting

How to write the `instructions` string for the design-capable model. The admin
Design tab (`src/admin/js/design-models.mjs`) encodes the same rules as a
per-model form with starter presets. Results vary run to run on every model:
compare several takes.

## Breeze-TTS-2 (`breeze-tts-2-mlx`)

By BreezeBlue (not MediaTek). Research / non-commercial licence.

- Free-text natural language. Match the instruction language to the text
  language (English or Chinese).
- Tags in the spoken text: English `(laugh)` `(cough)` `(clears throat)`
  `(sigh)`; Chinese `[笑]` `[咳嗽]` `[清嗓子]` `[叹气]`.
- Audio is capped at about 120 seconds.
- Examples: `A warm, thoughtful young woman with a clear voice and a calm, reflective delivery.`;
  `一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。`
