"""Unit tests for PyTorch-to-MLX decoder weight conversion (MLX is stubbed: no GPU needed)."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def _load():
    """Load dit_convert.py directly from its file."""
    path = Path(__file__).with_name("dit_convert.py")
    spec = importlib.util.spec_from_file_location("dit_convert_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dit_convert = _load()


class _Array:
    """Stand-in for mx.array that records the casts it goes through."""

    def __init__(self, value, dtype="float32"):
        self.value = value
        self.dtype = dtype

    def astype(self, dtype):
        return _Array(self.value, dtype)


def _fake_mx():
    """A minimal fake mlx.core that records every eval call."""
    mx = types.ModuleType("mlx.core")
    mx.evaluated = []
    mx.array = lambda value: _Array(value)
    mx.eval = lambda *arrays: mx.evaluated.extend(arrays)
    return mx


class _Decoder:
    """Decoder whose state_dict has a plain weight, a proj_in conv and a rotary buffer."""

    def state_dict(self):
        return {
            "layers.0.weight": torch.ones(2, 2),
            "proj_in.1.weight": torch.ones(2, 3, 4),
            "rotary_emb.inv_freq": torch.ones(2),
        }


class _Model:
    decoder = _Decoder()


class ConvertDecoderWeightsTests(unittest.TestCase):
    """Conversion keeps its behaviour and can cast one tensor at a time."""

    def convert(self, **kwargs):
        mx = _fake_mx()
        with patch.dict(sys.modules, {"mlx": types.ModuleType("mlx"), "mlx.core": mx}):
            return dit_convert.convert_decoder_weights(_Model(), **kwargs), mx

    def test_default_stays_float32_and_evaluates_nothing(self):
        """Without a dtype nothing is cast and no early eval happens (the old behaviour)."""
        weights, mx = self.convert()
        self.assertEqual({"layers.0.weight", "proj_in.weight"}, {k for k, _ in weights})
        self.assertTrue(all(a.dtype == "float32" for _, a in weights))
        self.assertEqual([], mx.evaluated)

    def test_dtype_casts_every_tensor_and_evaluates_each_one(self):
        """With a dtype each tensor is cast and evaluated right away, so the float32 original can be freed."""
        weights, mx = self.convert(dtype="bf16")
        self.assertTrue(all(a.dtype == "bf16" for _, a in weights))
        self.assertEqual(2, len(mx.evaluated))

    def test_layout_changes_are_unaffected_by_the_dtype(self):
        """proj_in conv weights are still swapped to [out, K, in] and rotary buffers still skipped."""
        weights, _ = self.convert(dtype="bf16")
        shapes = {k: a.value.shape for k, a in weights}
        self.assertEqual((2, 4, 3), shapes["proj_in.weight"])
        self.assertNotIn("rotary_emb.inv_freq", shapes)


if __name__ == "__main__":
    unittest.main()
