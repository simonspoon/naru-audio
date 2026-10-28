// Per-model data for the Design tab wizard, keyed by model id: which fields
// the model understands, how the form becomes the `instructions` string,
// starter presets, inline tags for the spoken text, tips and pitfalls.
// Unknown design models get GENERIC.
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
