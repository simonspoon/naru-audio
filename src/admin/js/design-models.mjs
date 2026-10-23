// Per-model data for the Design tab wizard, keyed by model id: which fields
// the model understands, how the form becomes the `instructions` string,
// starter presets, inline tags for the spoken text, tips and pitfalls.
// Unknown design models get GENERIC. Prompting notes behind this live in
// docs/voice-design-prompting.md.
//
// Field shapes:
//   { key, label, type: 'chips', options: [...], hint? }   one value, toggle
//   { key, label, type: 'select', options: [...], hint? }  one value, dropdown
//   { key, label, type: 'text', placeholder?, hint? }      free text
// A preset is { name, values, line?, text? }: `values` fills the form,
// `line` replaces the test line, `text` (when given) replaces the built
// prompt verbatim (used for hand-written, non-English instructions).

const GENDER = ['male', 'female', 'neutral'];
const AGE = ['child', 'teenager', 'young adult', 'middle-aged', 'senior'];
const PITCH = [
  'very low pitch',
  'low pitch',
  'slightly low pitch',
  'medium pitch',
  'slightly high pitch',
  'high pitch',
  'very high pitch',
];
const PACE = [
  'very slow pace',
  'slow pace',
  'slightly slow pace',
  'medium pace',
  'slightly fast pace',
  'fast pace',
  'very fast pace',
];
const EMOTION = [
  'cheerful', 'calm', 'gentle', 'serious', 'lively', 'composed', 'soothing', 'warm',
  'excited', 'sad', 'angry', 'menacing', 'playful', 'confident', 'melancholic',
  'mysterious', 'authoritative', 'tender', 'sarcastic', 'nervous', 'enthusiastic',
  'dreamy', 'weary',
];
const TIMBRE = [
  'resonant', 'crisp', 'husky', 'mellow', 'sweet', 'deep', 'powerful', 'smooth',
  'breathy', 'raspy', 'gravelly', 'silky', 'nasal', 'bright', 'velvety', 'airy',
];
const ACCENT = [
  'American accent', 'British accent', 'Australian accent', 'Canadian accent',
  'Indian accent', 'Irish accent', 'Scottish accent', 'Southern US accent',
  'French accent', 'German accent', 'Spanish accent', 'Japanese accent',
];
const SCENE = [
  'news broadcast', 'advertising', 'audiobook narration', 'animation character',
  'voice assistant', 'documentary narration', 'podcast host', 'meditation guide',
  'bedtime story', 'game character', 'customer service', 'radio DJ', 'museum guide',
  'sports commentary', 'movie trailer', 'classroom teacher',
];

const FIELD_GENDER = { key: 'gender', label: 'Gender', type: 'chips', options: GENDER };
const FIELD_AGE = { key: 'age', label: 'Age', type: 'chips', options: AGE };
const FIELD_PITCH = { key: 'pitch', label: 'Pitch', type: 'chips', options: PITCH };
const FIELD_PACE = { key: 'pace', label: 'Pace', type: 'chips', options: PACE };
const FIELD_EMOTION = { key: 'emotion', label: 'Tone', type: 'chips', options: EMOTION };
const FIELD_TIMBRE = { key: 'timbre', label: 'Timbre', type: 'chips', options: TIMBRE };
const FIELD_ACCENT = { key: 'accent', label: 'Accent', type: 'chips', options: ACCENT };
const FIELD_SCENE = { key: 'scene', label: 'Scene', type: 'chips', options: SCENE };
const FIELD_EXTRA = {
  key: 'extra',
  label: 'Extra',
  type: 'text',
  placeholder: 'anything else, e.g. like a fashion product presenter',
};

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

/** "Female, young adult, slightly high pitch, ..." (Qwen3-TTS style). */
function buildList(v) {
  const parts = [
    v.gender,
    v.age,
    v.pitch,
    v.pace,
    v.emotion && (v.emotion === 'cheerful' ? 'cheerful and lively' : `${v.emotion} tone`),
    v.timbre && `${v.timbre} timbre`,
    v.accent,
    v.scene && `suited to ${v.scene}`,
    v.extra && v.extra.trim(),
  ].filter(Boolean);
  return parts.length ? cap(parts.join(', ')) : '';
}

/** "A young adult female voice with a high pitch, ... tone." (open sentence). */
function buildSentence(v) {
  const subject = [v.age, v.gender].filter(Boolean).join(' ');
  const qualities = [
    v.pitch,
    v.pace,
    v.emotion && `${v.emotion} tone`,
    v.timbre && `${v.timbre} timbre`,
    v.accent,
    v.dialect && `speaking ${v.dialect}`,
  ].filter(Boolean);
  const any = subject || qualities.length || v.scene || (v.extra && v.extra.trim());
  if (!any) return '';
  let text = subject ? `A ${subject} voice` : 'A voice';
  if (qualities.length) text += ` with ${qualities.join(', ')}`;
  if (v.scene) text += `, suited to ${v.scene}`;
  if (v.extra && v.extra.trim()) text += `, ${v.extra.trim().replace(/\.$/, '')}`;
  return `${text}.`;
}

// Presets shared by the open-vocabulary models.
const OPEN_PRESETS = [
  { name: 'Documentary narrator', values: { gender: 'male', age: 'middle-aged', pitch: 'low pitch', pace: 'slow pace', emotion: 'calm', timbre: 'resonant', scene: 'documentary narration' } },
  { name: 'Cheerful teen', values: { gender: 'female', age: 'teenager', pitch: 'high pitch', pace: 'fast pace', emotion: 'cheerful', timbre: 'bright' } },
  { name: 'Wise elder', values: { gender: 'male', age: 'senior', pitch: 'low pitch', pace: 'slow pace', emotion: 'warm', timbre: 'mellow', extra: 'wise, like a grandfather telling stories' } },
  { name: 'Whisper ASMR', values: { gender: 'female', age: 'young adult', pitch: 'low pitch', pace: 'slow pace', emotion: 'soothing', timbre: 'breathy', extra: 'whispering softly close to the microphone' } },
  { name: 'News anchor', values: { gender: 'female', age: 'middle-aged', pitch: 'medium pitch', pace: 'medium pace', emotion: 'composed', timbre: 'crisp', scene: 'news broadcast' } },
  { name: 'Villain', values: { gender: 'male', age: 'middle-aged', pitch: 'very low pitch', pace: 'slow pace', emotion: 'menacing', timbre: 'raspy', scene: 'animation character' } },
  { name: 'Bedtime storyteller', values: { gender: 'female', age: 'middle-aged', pitch: 'slightly low pitch', pace: 'slow pace', emotion: 'gentle', timbre: 'mellow', scene: 'bedtime story' } },
  { name: 'Energetic ad announcer', values: { gender: 'male', age: 'young adult', pitch: 'slightly high pitch', pace: 'fast pace', emotion: 'enthusiastic', timbre: 'powerful', scene: 'advertising' } },
  { name: 'Voice assistant', values: { gender: 'neutral', age: 'young adult', pitch: 'medium pitch', pace: 'medium pace', emotion: 'calm', timbre: 'smooth', scene: 'voice assistant' } },
  { name: 'Excited kid', values: { gender: 'female', age: 'child', pitch: 'very high pitch', pace: 'fast pace', emotion: 'excited', timbre: 'sweet', scene: 'animation character' } },
];

const ZH_LINE = '今天天气真好，我们一起出去走走吧。';

// OmniVoice's closed vocabulary (k2-fsa): one value per category.
const OV = {
  gender: ['male', 'female'],
  age: ['child', 'teenager', 'young adult', 'middle-aged', 'elderly'],
  pitch: ['very low pitch', 'low pitch', 'moderate pitch', 'high pitch', 'very high pitch'],
  style: ['whisper'],
  accent: [
    'american accent', 'british accent', 'australian accent', 'canadian accent',
    'indian accent', 'chinese accent', 'korean accent', 'japanese accent',
    'portuguese accent', 'russian accent',
  ],
  dialect: [
    '河南话', '陕西话', '四川话', '贵州话', '云南话', '桂林话',
    '济南话', '石家庄话', '甘肃话', '宁夏话', '青岛话', '东北话',
  ],
};
// Order the categories appear in the built string.
const OV_ORDER = ['gender', 'age', 'pitch', 'accent', 'dialect', 'style'];

/** Maps each lowercased OmniVoice term to its category. */
const OV_CATEGORY = new Map(
  Object.entries(OV).flatMap(([cat, list]) => list.map((t) => [t.toLowerCase(), cat])),
);

function buildOmni(v) {
  const parts = OV_ORDER.map((k) => v[k]).filter(Boolean);
  // Half-width comma for English, full-width when a Chinese dialect is used.
  return parts.join(v.dialect ? '，' : ', ');
}

function checkOmni(text) {
  const warnings = [];
  const seen = new Map();
  for (const raw of text.split(/[,，]/)) {
    const term = raw.trim().toLowerCase();
    if (!term) continue;
    const cat = OV_CATEGORY.get(term);
    if (!cat) {
      warnings.push(`"${raw.trim()}" is not in OmniVoice's vocabulary and may be ignored or rejected.`);
      continue;
    }
    if (seen.has(cat)) warnings.push(`Two ${cat} values ("${seen.get(cat)}", "${term}"): use at most one per category.`);
    seen.set(cat, term);
  }
  if (seen.has('accent') && seen.has('dialect')) {
    warnings.push('Accent only affects English text and dialect only Chinese text; pick the one that matches your test line.');
  }
  return warnings;
}

const omniPreset = (name, values, line) => ({ name, values, ...(line ? { line } : {}) });

const OMNI = {
  label: 'OmniVoice',
  summary:
    'Closed vocabulary: pick at most one value per category. Anything outside the list is ignored or rejected, so there is no free text here.',
  license:
    'OmniVoice is released under CC-BY-NC: non-commercial use only. Do not use voices designed here in commercial products.',
  fields: [
    { key: 'gender', label: 'Gender', type: 'select', options: OV.gender },
    { key: 'age', label: 'Age', type: 'select', options: OV.age },
    { key: 'pitch', label: 'Pitch', type: 'select', options: OV.pitch },
    { key: 'accent', label: 'English accent', type: 'select', options: OV.accent, hint: 'English text only; no effect on Chinese.' },
    { key: 'dialect', label: 'Chinese dialect', type: 'select', options: OV.dialect, hint: 'Chinese text only; no effect on English.' },
    { key: 'style', label: 'Style', type: 'select', options: OV.style },
  ],
  build: buildOmni,
  check: checkOmni,
  placeholder: 'female, young adult, high pitch, british accent',
  tips: [
    'Comma-separated terms, case-insensitive, at most one per category. Partial strings like "female" are fine.',
    'Use a half-width comma for English and a full-width one (，) for Chinese.',
    'Voice design was trained on Chinese and English only.',
    'Insert tags such as [laughter] or [sigh] into the spoken text (the test line), not the voice description.',
  ],
  pitfalls: [
    'Accent has no effect on Chinese text; dialect has none on English text.',
    'Do not invent words ("raspy", "warm"): they are not in the vocabulary.',
  ],
  presets: [
    omniPreset('Documentary narrator', { gender: 'male', age: 'middle-aged', pitch: 'low pitch', accent: 'british accent' }),
    omniPreset('Cheerful teen', { gender: 'female', age: 'teenager', pitch: 'high pitch', accent: 'american accent' }),
    omniPreset('Wise elder', { gender: 'male', age: 'elderly', pitch: 'low pitch' }),
    omniPreset('Whisper ASMR', { gender: 'female', age: 'young adult', pitch: 'low pitch', style: 'whisper' }),
    omniPreset('News anchor', { gender: 'female', age: 'middle-aged', pitch: 'moderate pitch', accent: 'american accent' }),
    omniPreset('Villain', { gender: 'male', age: 'middle-aged', pitch: 'very low pitch', accent: 'british accent' }),
    omniPreset('Bedtime storyteller', { gender: 'female', age: 'middle-aged', pitch: 'low pitch', accent: 'canadian accent' }),
    omniPreset('Excited kid', { gender: 'female', age: 'child', pitch: 'very high pitch', accent: 'american accent' }),
    omniPreset('Sichuan chef', { gender: 'male', age: 'middle-aged', pitch: 'moderate pitch', dialect: '四川话' }, ZH_LINE),
    omniPreset('Northeastern uncle', { gender: 'male', age: 'middle-aged', pitch: 'low pitch', dialect: '东北话' }, ZH_LINE),
  ],
  tagLabel: 'Insert a tag into the test line',
  tags: [
    '[laughter]', '[sigh]', '[confirmation-en]', '[question-en]', '[question-ah]',
    '[question-oh]', '[question-ei]', '[question-yi]', '[surprise-ah]', '[surprise-oh]',
    '[surprise-wa]', '[surprise-yo]', '[dissatisfaction-hnn]',
  ].map((t) => ({ label: t, insert: t })),
};

const QWEN = {
  label: 'Qwen3-TTS VoiceDesign',
  summary:
    'Free-form natural language, kept separate from the spoken text. Combine several dimensions with concrete words.',
  charLimit: 500,
  fields: [
    FIELD_GENDER, FIELD_AGE, FIELD_PITCH, FIELD_PACE, FIELD_EMOTION, FIELD_TIMBRE,
    FIELD_ACCENT, FIELD_SCENE, FIELD_EXTRA,
  ],
  build: buildList,
  placeholder: 'Female, young adult, slightly high pitch, fast pace, cheerful and lively, like a fashion product presenter',
  tips: [
    'Combine gender, age, pitch, speed, emotion, timbre and scene in one line.',
    'Concrete words work best (resonant, crisp, husky, mellow); the Extra box takes full phrases such as "like a fashion product presenter".',
    'It can also take a direction, e.g. "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."',
    'Chinese and English are the benchmarked languages.',
  ],
  pitfalls: [
    'Keep it under about 500 characters (the hosted API limit; a soft guide for the open model).',
    'Avoid vague words ("nice", "normal"), "female voice" alone, and celebrity imitation.',
    'Accents are weak. Results vary run to run: generate a few takes and compare.',
    'Keep voice description out of the spoken text.',
  ],
  presets: OPEN_PRESETS,
  tags: [],
};

const VOX = {
  label: 'VoxCPM2',
  summary:
    'Open vocabulary. The server prepends your description to the text as "(description)text" itself, so write the bare description with no parentheses.',
  fields: [
    FIELD_GENDER, FIELD_AGE,
    { key: 'dialect', label: 'Dialect', type: 'chips', options: ['广东话', '四川话', '东北话', '河南话', '陕西话', '山东话', '上海话', '天津话', '闽南话'], hint: 'Also write the test line in that dialect wording.' },
    FIELD_PITCH, FIELD_PACE, FIELD_EMOTION, FIELD_TIMBRE, FIELD_SCENE, FIELD_EXTRA,
  ],
  build: buildSentence,
  placeholder: 'A warm, gentle female voice in her 30s with a calm and soothing tone',
  tips: [
    'Age, gender, pitch, pace, emotional tone and vocal texture all work; Chinese or English description.',
    'Supports 30 languages and several Chinese dialects: name the dialect in the description and write the text in dialect wording.',
    'Try 1 to 3 takes; results vary.',
  ],
  pitfalls: [
    'Do not type the parentheses yourself: they would be doubled on the wire.',
    'Very short test lines (under about 1 second of speech) sound weak.',
    'Very long text drifts: split it into shorter pieces.',
  ],
  presets: [
    ...OPEN_PRESETS,
    { name: 'Cantonese neighbour', values: { gender: 'female', age: 'middle-aged', emotion: 'warm', dialect: '广东话' }, line: '今日天气好好，不如出去饮杯茶啦。' },
  ],
  tags: [],
};

const BREEZE = {
  label: 'Breeze-TTS-2',
  summary:
    'Free-text natural language. Write the instruction in the same language as the spoken text (English or Chinese).',
  license:
    'Breeze-TTS-2 (by BreezeBlue) is research / non-commercial licensed. Check the licence before using generated voices commercially.',
  fields: [
    FIELD_GENDER, FIELD_AGE, FIELD_PITCH, FIELD_PACE, FIELD_EMOTION, FIELD_TIMBRE,
    FIELD_SCENE, FIELD_EXTRA,
  ],
  build: buildSentence,
  placeholder: 'A warm, thoughtful young woman with a clear voice and a calm, reflective delivery.',
  tips: [
    'Match the instruction language to the text language. The chips are English; for Chinese use a Chinese preset or type it.',
    'Tags go in the spoken text: (laugh) (cough) (clears throat) (sigh) in English, [笑] [咳嗽] [清嗓子] [叹气] in Chinese.',
    'Audio is capped at roughly 120 seconds.',
  ],
  pitfalls: [
    'An English instruction over Chinese text (or the reverse) gives weaker control.',
    'Put tags in the test line, not the voice description.',
  ],
  presets: [
    ...OPEN_PRESETS,
    { name: '温柔年轻女性 (中文)', values: {}, text: '一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。', line: ZH_LINE },
    { name: '沉稳中年男性 (中文)', values: {}, text: '一位沉稳的中年男性，声音低沉厚重，语速缓慢，语气平静而有说服力。', line: ZH_LINE },
  ],
  tagLabel: 'Insert a tag into the test line',
  tags: [
    { label: '(laugh)', insert: '(laugh)' },
    { label: '(cough)', insert: '(cough)' },
    { label: '(clears throat)', insert: '(clears throat)' },
    { label: '(sigh)', insert: '(sigh)' },
    { label: '[笑]', insert: '[笑]' },
    { label: '[咳嗽]', insert: '[咳嗽]' },
    { label: '[清嗓子]', insert: '[清嗓子]' },
    { label: '[叹气]', insert: '[叹气]' },
  ],
};

const GENERIC = {
  label: 'Design model',
  summary: 'No model-specific notes; the description is sent as free text.',
  fields: [
    FIELD_GENDER, FIELD_AGE, FIELD_PITCH, FIELD_PACE, FIELD_EMOTION, FIELD_TIMBRE,
    FIELD_ACCENT, FIELD_SCENE, FIELD_EXTRA,
  ],
  build: buildSentence,
  placeholder: 'A cheerful young female voice with high pitch',
  tips: ['Combine a few concrete attributes: gender, age, pitch, pace, tone.'],
  pitfalls: ['Results vary between takes; compare all three.'],
  presets: OPEN_PRESETS,
  tags: [],
};

export const DESIGN_MODELS = {
  'qwen3-tts-1.7b-voicedesign-mlx': QWEN,
  'voxcpm2-8bit-mlx': VOX,
  'omnivoice-bf16-mlx': OMNI,
  'breeze-tts-2-mlx': BREEZE,
  'breeze-tts-2-8bit-mlx': BREEZE,
  'breeze-tts-2-4bit-mlx': BREEZE,
};

/** Model-specific data for `id`, or the generic fallback. `hint` (the
 * manifest's own `x_prompt_format.hint`) becomes the fallback placeholder. */
export function designModel(id, hint) {
  const known = DESIGN_MODELS[id];
  if (known) return known;
  return hint ? { ...GENERIC, placeholder: hint } : GENERIC;
}
