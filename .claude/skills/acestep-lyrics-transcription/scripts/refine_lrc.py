#!/usr/bin/env python3
"""Refine an ACE-Step API LRC against the isolated vocal stem.

The DiT alignment returns the right text, but it also times lines the model never
sang and pins the first line to 0:00 during an instrumental intro. This script
detects where vocals actually sound (ffmpeg silencedetect) and then:
  * snaps a line that starts in silence forward to the next vocal onset,
  * drops a line whose time window holds no vocals at all,
  * strips section tags such as [Verse] (unless --keep-tags),
  * drops lines stacked on one moment (they could never be read) and writes an empty-text
    timestamp where the vocals stop, the standard LRC way to end a line.

Usage: refine_lrc.py <audio> <lrc> [--vocals stem.mp3] [--out refined.lrc]
       [--noise-db -38] [--min-silence 0.5] [--keep-tags]
Without --vocals, demucs separates the vocals first (uv tool install demucs).
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

LINE_RE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")
TAG_RE = re.compile(r"^\[[^\]]+\]$")
SILENCE_RE = re.compile(r"silence_(start|end): (-?\d+(?:\.\d+)?)")


def parse_lrc(text):
    """Return a list of (seconds, text) for every timestamped LRC line."""
    lines = []
    for raw in text.splitlines():
        m = LINE_RE.match(raw.strip())
        if m:
            lines.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
    return lines


def format_lrc(lines):
    """Format (seconds, text) pairs as LRC text."""
    return "".join(f"[{int(t // 60):02d}:{t % 60:05.2f}]{x}\n" for t, x in lines)


def sound_intervals(silences, duration):
    """Convert ffmpeg (kind, time) silence events into (start, end) sound intervals."""
    sound, cursor = [], 0.0
    for kind, t in silences:
        if kind == "start":
            if t > cursor:
                sound.append((cursor, t))
        else:
            cursor = max(cursor, t)
    if cursor < duration:
        sound.append((cursor, duration))
    return sound


def drop_blips(sound, min_length=0.6):
    """Ignore stretches of "sound" shorter than min_length: separation leaks (a hi-hat, a snare) are not singing."""
    return [(s, e) for s, e in sound if e - s >= min_length]


def refine_lines(lines, sound, duration, keep_tags=False):
    """Snap lines to vocal onsets and drop lines without vocals.

    Returns (kept_lines, dropped_lines, moved) where moved lists (old, new, text).
    """
    kept, dropped, moved = [], [], []
    for i, (start, text) in enumerate(lines):
        if TAG_RE.match(text) or not text:
            if keep_tags:
                kept.append((start, text))
            continue
        end = lines[i + 1][0] if i + 1 < len(lines) else duration
        overlap = [(max(s, start), e) for s, e in sound if e > start and s < end]
        if not overlap:
            dropped.append((start, text))
            continue
        new_start = overlap[0][0]
        if new_start - start > 0.05:
            moved.append((start, new_start, text))
        kept.append((new_start, text))
    return kept, dropped, moved


def drop_stacked(kept, min_gap=0.35):
    """Drop lines that start less than min_gap before the next line: they could never be read.

    The DiT aligner crams lines the model never sang onto one timestamp; keeping only the last of
    such a stack avoids 0-second flashes. Returns (kept, dropped).
    """
    keep, dropped = [], []
    for i, (start, text) in enumerate(kept):
        if i + 1 < len(kept) and kept[i + 1][0] - start < min_gap:
            dropped.append((start, text))
        else:
            keep.append((start, text))
    return keep, dropped


def end_markers(kept, sound, duration, min_gap=1.0):
    """Empty-text timestamps (standard LRC "line ends here") where the vocals stop for good.

    A marker is placed at the end of the vocal stretch a line starts in, but only if the next line
    begins more than min_gap later (or there is no next line), so a pause between two lines hides
    the old line and a long silence after the last line does not leave it on screen.
    """
    markers = []
    for i, (start, _text) in enumerate(kept):
        nxt = kept[i + 1][0] if i + 1 < len(kept) else duration
        stretch_end = next((e for s, e in sound if s <= start + 0.05 < e), None)
        if stretch_end is not None and nxt - stretch_end > min_gap:
            markers.append((stretch_end, ""))
    return markers


def detect_silences(vocals, noise_db, min_silence):
    """Run ffmpeg silencedetect and return ((kind, time) events, duration)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", vocals],
        capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", vocals, "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL)
    events = [(k, float(v)) for k, v in SILENCE_RE.findall(proc.stderr)]
    return events, float(probe.stdout.strip())


def separate_vocals(audio, workdir):
    """Separate vocals with demucs and return the vocals stem path."""
    # --shifts 0: no random time shifts, so the same file always gives the same vocal stem (the default is random)
    subprocess.run(["demucs", "--two-stems=vocals", "--shifts", "0", "-o", workdir, "--mp3", audio],
                   check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return str(Path(workdir) / "htdemucs" / Path(audio).stem / "vocals.mp3")


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("audio")
    ap.add_argument("lrc")
    ap.add_argument("--vocals")
    ap.add_argument("--out")
    ap.add_argument("--noise-db", type=float, default=-38)
    ap.add_argument("--min-silence", type=float, default=0.5)
    ap.add_argument("--min-sound", type=float, default=0.6, help="ignore sound shorter than this (seconds)")
    ap.add_argument("--keep-tags", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        vocals = args.vocals or separate_vocals(args.audio, tmp)
        events, duration = detect_silences(vocals, args.noise_db, args.min_silence)
    lines = parse_lrc(Path(args.lrc).read_text(encoding="utf-8"))
    sound = drop_blips(sound_intervals(events, duration), args.min_sound)
    kept, dropped, moved = refine_lines(lines, sound, duration, args.keep_tags)
    kept, stacked = drop_stacked(kept)
    markers = end_markers(kept, sound, duration)
    result = sorted(kept + markers, key=lambda x: (x[0], x[1] == ""))

    out = args.out or str(Path(args.lrc).with_suffix(".refined.lrc"))
    Path(out).write_text(format_lrc(result), encoding="utf-8")
    for old, new, text in moved:
        print(f"moved   {old:6.2f} -> {new:6.2f}  {text}", file=sys.stderr)
    for start, text in dropped:
        print(f"dropped {start:6.2f}  {text}", file=sys.stderr)
    for start, text in stacked:
        print(f"stacked {start:6.2f}  {text}  (same moment as a later line)", file=sys.stderr)
    for t, _ in markers:
        print(f"ends    {t:6.2f}  (vocals stop)", file=sys.stderr)
    print(out)


if __name__ == "__main__":
    main()
