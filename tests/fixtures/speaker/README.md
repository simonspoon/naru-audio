# Speaker fixtures (naru task 2028)

16 kHz mono 16-bit WAVs generated on macOS with `say` + `afconvert` (2026-10-18):

    say -v <voice> -o x.aiff "<text>"; afconvert -f WAVE -d LEI16@16000 -c 1 x.aiff x.wav

| file | voice | text |
|---|---|---|
| `a1.wav` | Daniel (en_GB, male) | The quarterly report is ready for review, and I will send it over this afternoon. |
| `a2.wav` | Daniel | Please move the meeting to Thursday morning, because the team will be travelling on Wednesday. |
| `a3.wav` | Daniel | I think we should review the budget again before the end of the month. |
| `a_short.wav` | Daniel | Yes. |
| `b1.wav`, `b2.wav`, `b3.wav` | Samantha (en_US, female) | the same three sentences as a1, a2, a3 |

Voice A is the enrolled speaker (`a1`, `a2`; `a3` is held out), B the other. These are clean synthetic voices, so
scores are cleaner than real speech: the threshold was set on real read speech (`src/speaker.rs`,
`DEFAULT_THRESHOLD`); re-measure it against a real enrollment.
