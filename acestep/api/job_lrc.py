"""Optional LRC (timed lyrics) generation for finished API generation jobs."""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger

_REQUIRED_KEYS = (
    "pred_latents",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "context_latents",
    "lyric_token_idss",
)
_LATENT_FRAMES_PER_SECOND = 25.0


def _missing_tensor_keys(extra_outputs: dict[str, Any]) -> list[str]:
    """Return required alignment tensor keys absent from ``extra_outputs``."""

    return [key for key in _REQUIRED_KEYS if extra_outputs.get(key) is None]


def _align_one(
    *,
    dit_handler: Any,
    extra_outputs: dict[str, Any],
    index: int,
    audio_duration: float,
    vocal_language: str,
    inference_steps: int,
) -> str:
    """Align lyrics to one generated sample and return its LRC text ("" on failure)."""

    sl = slice(index, index + 1)
    result = dit_handler.get_lyric_timestamp(
        pred_latent=extra_outputs["pred_latents"][sl],
        encoder_hidden_states=extra_outputs["encoder_hidden_states"][sl],
        encoder_attention_mask=extra_outputs["encoder_attention_mask"][sl],
        context_latents=extra_outputs["context_latents"][sl],
        lyric_token_ids=extra_outputs["lyric_token_idss"][sl],
        total_duration_seconds=float(audio_duration),
        vocal_language=vocal_language or "en",
        inference_steps=int(inference_steps),
        seed=42,
    )
    if not result.get("success"):
        logger.warning(f"[api-lrc] alignment failed for sample {index}: {result.get('error')}")
        return ""
    return result.get("lrc_text", "") or ""


def build_lrcs_for_result(
    *,
    dit_handler: Any,
    extra_outputs: dict[str, Any],
    num_audios: int,
    audio_duration: float | None,
    vocal_language: str,
    inference_steps: int,
    align_fn: Callable[..., str] = _align_one,
) -> list[str]:
    """Build one LRC string per generated audio; failures yield empty strings.

    Never raises: alignment is an optional extra and must not fail the job.

    Args:
        dit_handler: DiT handler exposing ``get_lyric_timestamp``.
        extra_outputs: ``GenerationResult.extra_outputs`` holding the generation tensors.
        num_audios: Number of generated audios in the result.
        audio_duration: Requested duration in seconds; derived from latents when unset or <= 0.
        vocal_language: Language code of the lyrics.
        inference_steps: Diffusion steps used for the generation.
        align_fn: Per-sample alignment function (injectable for tests).

    Returns:
        A list of ``num_audios`` LRC strings.
    """

    empty = [""] * num_audios
    missing = _missing_tensor_keys(extra_outputs or {})
    if missing:
        logger.warning(f"[api-lrc] skipped, generation tensors unavailable: {missing}")
        return empty

    batch = extra_outputs["pred_latents"].shape[0]
    duration = audio_duration
    if duration is None or duration <= 0:
        duration = extra_outputs["pred_latents"].shape[1] / _LATENT_FRAMES_PER_SECOND

    lrcs = list(empty)
    for index in range(min(num_audios, batch)):
        try:
            lrcs[index] = align_fn(
                dit_handler=dit_handler,
                extra_outputs=extra_outputs,
                index=index,
                audio_duration=duration,
                vocal_language=vocal_language,
                inference_steps=inference_steps,
            )
        except Exception:  # noqa: BLE001 - optional feature must not fail the job
            logger.exception(f"[api-lrc] unexpected error aligning sample {index}")
    return lrcs
