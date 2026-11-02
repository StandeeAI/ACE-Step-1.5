"""Time lyric lines from what is really sung: Whisper word times on the vocal stem, matched to the lyrics.

The DiT aligner of ACE-Step times lines roughly and crams unsung lines together. Here each line gets the
time at which its first word is heard, and lines that are not heard are dropped. Stdlib only (Whisper is
called through the `mlx_whisper` command when it is installed).
"""
import difflib
import json
import re
import subprocess
from pathlib import Path

MODEL = "mlx-community/whisper-large-v3-turbo"


def normalize(text):
    """Lowercase words without punctuation: "Dance, with me!" -> ["dance", "with", "me"]."""
    return [w for w in re.sub(r"[^a-z0-9' ]+", " ", text.lower()).replace("'", "").split() if w]


def same_word(a, b):
    """Equal, or close enough for a misheard word (long words only)."""
    return a == b or (min(len(a), len(b)) >= 4 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8)


def heard_in_sound(words, sound, margin=0.4):
    """Keep Whisper words that overlap a stretch where the stem really sounds (drops "Thank you" hallucinated in silence)."""
    return [w for w in words if any(w[1] < e + margin and w[2] > s - margin for s, e in sound)]


def match_words(line_tokens, words):
    """Monotonic fuzzy alignment (longest common subsequence). Returns {line_token_index: word_index}."""
    n, m = len(line_tokens), len(words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            hit = same_word(line_tokens[i - 1][1], words[j - 1][0])
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + (1 if hit else 0))
    pairs, i, j = {}, n, m
    while i > 0 and j > 0:
        if same_word(line_tokens[i - 1][1], words[j - 1][0]) and dp[i][j] == dp[i - 1][j - 1] + 1:
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return pairs


def time_lines(texts, words, word_gap=0.3):
    """Start/end per lyric line from the heard words.

    texts: lyric lines; words: [(word, start, end)] in time order.
    Returns (timed, unheard): timed = [(start, end, text)], unheard = [text] of lines with too few heard words.
    """
    tokens = [(li, tok) for li, text in enumerate(texts) for tok in normalize(text)]
    pairs = match_words(tokens, words)
    timed, unheard = [], []
    for li, text in enumerate(texts):
        idx = [k for k, (l, _t) in enumerate(tokens) if l == li]
        hit = [k for k in idx if k in pairs]
        if not idx or len(hit) < max(1, -(-len(idx) // 2)):
            unheard.append(text)
            continue
        lead, trail = hit[0] - idx[0], idx[-1] - hit[-1]
        start = words[pairs[hit[0]]][1] - word_gap * lead
        end = words[pairs[hit[-1]]][2] + word_gap * trail
        timed.append((max(0.0, start), end, text))
    return timed, unheard


def transcribe(vocals, workdir, language="en"):
    """Whisper word times [(word, start, end)] for a vocal stem, or None when mlx_whisper is not installed."""
    try:
        subprocess.run(
            ["mlx_whisper", vocals, "--model", MODEL, "--language", language, "-f", "json", "--output-dir", workdir,
             "--output-name", "words", "--word-timestamps", "True", "--condition-on-previous-text", "False"],
            check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    data = json.loads((Path(workdir) / "words.json").read_text(encoding="utf-8"))
    return [(normalize(w["word"])[0] if normalize(w["word"]) else "", w["start"], w["end"])
            for s in data["segments"] for w in s.get("words", []) if normalize(w["word"])]
