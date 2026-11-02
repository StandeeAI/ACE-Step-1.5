"""Unit tests for the LRC refinement helpers (pure functions, no audio needed)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from refine_lrc import format_lrc, parse_lrc, refine_lines, sound_intervals  # noqa: E402


class RefineLrcTests(unittest.TestCase):
    """Behavior tests for snapping and dropping LRC lines."""

    def test_parse_and_format_round_trip(self):
        """Timestamps survive parse then format."""

        lines = parse_lrc("[00:01.50]hello\n[01:02.00]world\nnot a line\n")
        self.assertEqual([(1.5, "hello"), (62.0, "world")], lines)
        self.assertEqual("[00:01.50]hello\n[01:02.00]world\n", format_lrc(lines))

    def test_sound_intervals_from_silences(self):
        """Silence events invert into sound intervals, including leading and trailing."""

        events = [("start", 0.0), ("end", 17.0), ("start", 30.0), ("end", 40.0)]
        self.assertEqual([(17.0, 30.0), (40.0, 60.0)], sound_intervals(events, 60.0))

    def test_first_line_snaps_to_vocal_onset(self):
        """A line pinned to 0:00 during the intro moves to the first vocal onset."""

        lines = [(0.24, "[Verse]"), (0.24, "Baila"), (20.0, "Feel the night")]
        kept, dropped, moved = refine_lines(lines, [(17.0, 30.0)], 60.0)
        self.assertEqual([(17.0, "Baila"), (20.0, "Feel the night")], kept)
        self.assertEqual([], dropped)
        self.assertEqual([(0.24, 17.0, "Baila")], moved)

    def test_line_without_vocals_is_dropped(self):
        """A line whose window holds only silence is removed."""

        lines = [(5.0, "sung"), (30.0, "never sung"), (50.0, "sung again")]
        sound = [(4.0, 20.0), (50.0, 60.0)]
        kept, dropped, _ = refine_lines(lines, sound, 60.0)
        self.assertEqual([(5.0, "sung"), (50.0, "sung again")], kept)
        self.assertEqual([(30.0, "never sung")], dropped)

    def test_tags_kept_only_on_request(self):
        """Section tags are stripped by default and kept with keep_tags."""

        lines = [(1.0, "[Chorus]"), (1.0, "la")]
        self.assertEqual([(1.0, "la")], refine_lines(lines, [(0.0, 5.0)], 5.0)[0])
        self.assertEqual(2, len(refine_lines(lines, [(0.0, 5.0)], 5.0, keep_tags=True)[0]))


if __name__ == "__main__":
    unittest.main()
