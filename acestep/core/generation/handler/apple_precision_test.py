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


if __name__ == "__main__":
    unittest.main()
