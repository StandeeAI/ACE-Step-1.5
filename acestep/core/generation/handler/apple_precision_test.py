"""Unit tests for the Apple Silicon precision and memory-release helpers."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


def _load():
    """Load apple_precision.py directly (it has no package-level dependencies)."""
    path = Path(__file__).with_name("apple_precision.py")
    spec = importlib.util.spec_from_file_location("apple_precision_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


precision = _load()


class NonCudaDtypeTests(unittest.TestCase):
    """The default dtype for non-CUDA devices."""

    def test_defaults_are_unchanged(self):
        """MPS and CPU stay on float32 and XPU on bfloat16 unless something opts in."""
        self.assertEqual(torch.float32, precision.non_cuda_dtype("mps", env={}))
        self.assertEqual(torch.float32, precision.non_cuda_dtype("cpu", env={}))
        self.assertEqual(torch.bfloat16, precision.non_cuda_dtype("xpu", env={}))

    def test_mps_bf16_is_opt_in(self):
        """ACESTEP_MPS_BF16 switches MPS to bfloat16 and leaves other devices alone."""
        env = {"ACESTEP_MPS_BF16": "1"}
        self.assertEqual(torch.bfloat16, precision.non_cuda_dtype("mps", env=env))
        self.assertEqual(torch.float32, precision.non_cuda_dtype("cpu", env=env))

    def test_falsy_values_do_not_enable_it(self):
        """'0', 'false', empty and junk all mean off."""
        for value in ("0", "false", "", "no", "banana"):
            self.assertFalse(precision.env_flag("X", {"X": value}), value)
        for value in ("1", "true", "YES", " on "):
            self.assertTrue(precision.env_flag("X", {"X": value}), value)


class ReleaseMemoryTests(unittest.TestCase):
    """Emptying the allocator caches after loading."""

    def test_only_mps_is_touched(self):
        """On other devices nothing is called."""
        with patch.object(precision.torch.mps, "empty_cache") as empty:
            precision.release_unused_device_memory("cuda")
            precision.release_unused_device_memory("cpu")
        empty.assert_not_called()

    def test_mps_empties_both_caches(self):
        """MPS: PyTorch's cache and MLX's cache are both emptied."""
        fake_mx = MagicMock()
        fake_mlx = MagicMock()
        fake_mlx.core = fake_mx  # `import mlx.core as mx` reads the attribute of the package
        with patch.object(precision.torch.mps, "empty_cache") as empty, patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_mx}):
            precision.release_unused_device_memory("mps")
        empty.assert_called_once()
        fake_mx.clear_cache.assert_called_once()

    def test_missing_mlx_is_not_an_error(self):
        """Without MLX installed the PyTorch cache is still emptied and nothing raises."""
        with patch.object(precision.torch.mps, "empty_cache") as empty, patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
            precision.release_unused_device_memory("mps")
        empty.assert_called_once()


class _FakeDecoder:
    """Records where it was moved."""

    def __init__(self):
        self.moves = []

    def to(self, target):
        self.moves.append(target)
        return self


class _FakeHandler:
    """Minimal handler state used by settle_pending_decoder."""

    def __init__(self, decoder=None, use_mlx=False, mlx_decoder=None):
        self.model = type("Model", (), {"decoder": None})()
        self._pending_decoder = decoder
        self.use_mlx_dit = use_mlx
        self.mlx_decoder = mlx_decoder
        self.dtype = torch.bfloat16


class FreePytorchDecoderTests(unittest.TestCase):
    """Holding the PyTorch decoder on the CPU and dropping it once MLX took over."""

    def test_flag_is_opt_in(self):
        """Off unless ACESTEP_FREE_PYTORCH_DECODER is truthy."""
        self.assertFalse(precision.free_pytorch_decoder_requested({}))
        self.assertTrue(precision.free_pytorch_decoder_requested({"ACESTEP_FREE_PYTORCH_DECODER": "1"}))

    def test_detach_decoder_leaves_none_and_returns_it(self):
        """The model keeps no decoder; the caller gets it."""
        model = type("Model", (), {"decoder": object()})()
        decoder = model.decoder
        self.assertIs(decoder, precision.detach_decoder(model))
        self.assertIsNone(model.decoder)

    def test_settle_drops_the_cpu_decoder_when_mlx_took_over(self):
        """MLX decoder active: nothing is put back, the pending copy is released."""
        handler = _FakeHandler(decoder=_FakeDecoder(), use_mlx=True, mlx_decoder=object())
        precision.settle_pending_decoder(handler, "mps")
        self.assertIsNone(handler._pending_decoder)
        self.assertIsNone(handler.model.decoder)

    def test_settle_restores_the_decoder_when_mlx_failed(self):
        """No MLX decoder: the PyTorch decoder goes back to the device in the handler dtype, so generation still works."""
        decoder = _FakeDecoder()
        handler = _FakeHandler(decoder=decoder, use_mlx=False)
        precision.settle_pending_decoder(handler, "mps")
        self.assertIs(decoder, handler.model.decoder)
        self.assertEqual(["mps", torch.bfloat16], decoder.moves)

    def test_settle_without_a_pending_decoder_does_nothing(self):
        """The normal path is untouched."""
        handler = _FakeHandler(decoder=None)
        precision.settle_pending_decoder(handler, "mps")
        self.assertIsNone(handler.model.decoder)


if __name__ == "__main__":
    unittest.main()
