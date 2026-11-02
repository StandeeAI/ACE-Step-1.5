"""Unit tests for the Whisper-based lyric timing (pure functions, no audio, no Whisper)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lrc_align import heard_in_sound, match_words, normalize, time_lines  # noqa: E402

WORDS = [("tangerine", 17.6, 18.8), ("midnight", 19.0, 19.9), ("neon", 20.6, 21.0), ("on", 21.6, 21.7),
         ("my", 21.7, 22.0), ("skin", 22.1, 22.9), ("let", 28.6, 29.0), ("the", 29.6, 29.8), ("music", 29.9, 30.3), ("in", 30.5, 30.9)]
LINES = ["Tangerine midnight", "Neon on my skin", "Feel the lights", "Let the music in", "All night long"]


class AlignTests(unittest.TestCase):
    """Lines get the time of their first heard word."""

    def test_normalize(self):
        """Punctuation and case do not matter."""
        self.assertEqual(["dance", "with", "me"], normalize("Dance, with me!"))
        self.assertEqual(["dont", "stop"], normalize("Don't stop"))

    def test_each_line_starts_where_its_first_word_is_heard(self):
        """The first line starts at 17.6 s, not where the aligner guessed."""
        timed, _ = time_lines(LINES, WORDS)
        self.assertEqual([("Tangerine midnight", 17.6), ("Neon on my skin", 20.6), ("Let the music in", 28.6)],
                         [(t, s) for s, _e, t in timed][:3])

    def test_unheard_lines_are_dropped(self):
        """Lines the singer never sang are reported, not shown."""
        _, unheard = time_lines(LINES, WORDS)
        self.assertEqual(["Feel the lights", "All night long"], unheard)

    def test_repeated_phrases_follow_the_order_of_the_song(self):
        """Two identical lines map to the first and second time they are heard."""
        words = [("dance", 33.9, 34.2), ("with", 35.0, 35.2), ("me", 35.4, 35.8), ("dance", 42.6, 42.9), ("with", 43.7, 43.9), ("me", 43.9, 44.3)]
        timed, _ = time_lines(["Dance with me", "Dance with me"], words)
        self.assertEqual([33.9, 42.6], [s for s, _e, _t in timed])

    def test_misheard_word_still_matches(self):
        """A close spelling of a long word counts as heard."""
        self.assertEqual({0: 0, 1: 1}, match_words([(0, "tangerine"), (0, "midnight")], [("tangerines", 1, 2), ("midnight", 2, 3)]))

    def test_one_matching_word_is_not_enough_for_a_long_line(self):
        """A line with 6 words of which 1 matches is not considered sung."""
        timed, unheard = time_lines(["the quick brown fox jumps over"], [("the", 5.0, 5.2)])
        self.assertEqual(([], ["the quick brown fox jumps over"]), (timed, unheard))

    def test_hallucination_in_silence_is_filtered(self):
        """Whisper words where the stem is silent ("Thank you" at 58 s) are dropped."""
        words = [("neon", 20.6, 21.0), ("thank", 58.1, 58.5)]
        self.assertEqual([("neon", 20.6, 21.0)], heard_in_sound(words, [(18.0, 23.0)]))


if __name__ == "__main__":
    unittest.main()
