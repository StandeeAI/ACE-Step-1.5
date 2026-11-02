---
name: acestep-vocal-polish
description: Make ACE-Step vocals sound less synthetic and fix their level and tone. Separates vocals with demucs, treats them with ffmpeg (EQ, compression, echo, vinyl noise) and remixes. Use when users say the vocals sound AI, too quiet or loud, hollow or harsh.
allowed-tools: Read, Write, Bash
---

# Vocal Polish Skill

Post-process generated tracks so the vocals sit in the mix and sound less AI.

## Requirements

- `ffmpeg` and `ffprobe` on PATH.
- `demucs`, installed in its own environment (not in the repo):
  `uv tool install demucs --python 3.11 --with numpy --with soundfile`
  (without `numpy` demucs fails with `No module named 'numpy'`).

## Usage

```bash
cd "{project_root}/{.claude or .codex}/skills/acestep-vocal-polish/"
bash ./scripts/polish-vocals.sh <track.mp3> [--preset subtle|vintage] [--out FILE] \
    [--vocal-gain 1.65] [--music-gain 0.85] [--echo 0.14] [--lufs -15.5]
```

Output defaults to `polished-<preset>_<name>.mp3` next to the input. A 60 s track
separates in about 10 s on Apple Silicon.

## Presets

- `subtle` (default): high-pass 110 Hz, low-pass 9.5 kHz, compression, +2 dB at
  250 Hz, cuts at 2.8 kHz and 5.5 kHz, one short quiet echo, faint vinyl noise
  and tape wobble. Judged best by ear.
- `vintage`: narrow band (220 Hz to 7 kHz) and heavier saturation. It tends to sound hollow.

## Tuning by symptom

| Complaint | Change |
|---|---|
| Vocals too quiet / loud | `--vocal-gain` up / down (about 1 to 2.5), or `--music-gain` |
| Hollow | Lower `--echo` (try 0.08, or 0). Several echoes stacked cause comb filtering |
| Harsh / shrill | Lower the 5.5 kHz and 2.8 kHz bands in the script, lower the low-pass. Never boost 3 kHz |
| Dull | Raise the low-pass or soften the 5.5 kHz cut |
| Whole track too quiet | `--lufs` toward -14 |

Change one thing per iteration (about 1.5 dB steps) and re-listen.

## Getting better raw vocals first

Post-processing only hides artefacts, so also improve the source:

- Caption: "breathy female vocal, close-mic, doubled backing vocals, plate reverb,
  tape saturation, lo-fi vinyl, vintage analog recording".
- Lyrics: real short words ("baila", "move your body") rather than only "ooh-ooh".
- Generate several variants (`batch_size`), pick the best vocal, then polish it.
- A larger quality model (sft/base, more steps) helps more than turbo.

## Pitfalls

- A reference audio via REST must be uploaded as multipart (`ref_audio=@file`).
  `reference_audio_path` rejects absolute paths outside the temp directory.
- Reference clips: cut a 30 s excerpt with ffmpeg instead of a long mix.
- In shell loops, give `ffmpeg` and `demucs` `</dev/null` or they eat the loop's stdin.
- The same seed with a different duration does not reproduce the same song.
