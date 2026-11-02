"""Unit tests for optional LRC generation helpers in API jobs."""

import unittest
from unittest.mock import MagicMock

from acestep.api.job_lrc import build_lrcs_for_result


class _Tensor:
    """Tiny tensor stand-in supporting shape and batch slicing."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Store the tensor shape."""

        self.shape = shape

    def __getitem__(self, item):
        """Return a one-row tensor for a slice over the batch dimension."""

        return _Tensor((1,) + self.shape[1:])


def _extra_outputs(batch: int = 2, frames: int = 4500) -> dict:
    """Build the extra_outputs mapping that alignment requires."""

    return {
        "pred_latents": _Tensor((batch, frames, 64)),
        "encoder_hidden_states": _Tensor((batch, 10, 8)),
        "encoder_attention_mask": _Tensor((batch, 10)),
        "context_latents": _Tensor((batch, frames, 128)),
        "lyric_token_idss": _Tensor((batch, 20)),
    }


class BuildLrcsForResultTests(unittest.TestCase):
    """Behavior tests for per-audio LRC generation."""

    def test_returns_one_lrc_per_audio(self):
        """Each generated audio should get the LRC text from the handler."""

        handler = MagicMock()
        handler.get_lyric_timestamp.return_value = {"success": True, "lrc_text": "[00:01.00]hi"}
        lrcs = build_lrcs_for_result(
            dit_handler=handler,
            extra_outputs=_extra_outputs(),
            num_audios=2,
            audio_duration=60.0,
            vocal_language="en",
            inference_steps=8,
        )
        self.assertEqual(["[00:01.00]hi", "[00:01.00]hi"], lrcs)
        kwargs = handler.get_lyric_timestamp.call_args.kwargs
        self.assertEqual(60.0, kwargs["total_duration_seconds"])
        self.assertEqual(8, kwargs["inference_steps"])

    def test_missing_tensors_yield_empty_strings(self):
        """Without generation tensors (e.g. save-memory mode) no alignment is attempted."""

        handler = MagicMock()
        lrcs = build_lrcs_for_result(
            dit_handler=handler,
            extra_outputs={"pred_latents": None},
            num_audios=2,
            audio_duration=30.0,
            vocal_language="en",
            inference_steps=8,
        )
        self.assertEqual(["", ""], lrcs)
        handler.get_lyric_timestamp.assert_not_called()

    def test_handler_failure_and_exception_do_not_raise(self):
        """A failed or crashing alignment must leave an empty LRC, not fail the job."""

        handler = MagicMock()
        handler.get_lyric_timestamp.side_effect = [
            {"success": False, "error": "boom"},
            RuntimeError("unexpected"),
        ]
        lrcs = build_lrcs_for_result(
            dit_handler=handler,
            extra_outputs=_extra_outputs(),
            num_audios=2,
            audio_duration=30.0,
            vocal_language="en",
            inference_steps=8,
        )
        self.assertEqual(["", ""], lrcs)

    def test_duration_is_derived_from_latents_when_unset(self):
        """A non-positive duration falls back to latent frames at 25 per second."""

        handler = MagicMock()
        handler.get_lyric_timestamp.return_value = {"success": True, "lrc_text": "x"}
        build_lrcs_for_result(
            dit_handler=handler,
            extra_outputs=_extra_outputs(batch=1, frames=1500),
            num_audios=1,
            audio_duration=-1,
            vocal_language="",
            inference_steps=8,
        )
        kwargs = handler.get_lyric_timestamp.call_args.kwargs
        self.assertEqual(60.0, kwargs["total_duration_seconds"])
        self.assertEqual("en", kwargs["vocal_language"])


if __name__ == "__main__":
    unittest.main()
