"""Unit tests for model architecture and parameter count."""

import unittest
import torch

from simplexity.model import (
    SimplexityModelConfig,
    SimplexityTransformer,
    count_parameters,
)


class TestModel(unittest.TestCase):
    def test_mess3_parameter_count(self):
        cfg = SimplexityModelConfig(d_vocab=3)
        model = SimplexityTransformer(cfg)
        params = count_parameters(model)
        self.assertEqual(params["total"], 143_075)
        self.assertEqual(params["trainable"], 143_075)

    def test_rrxor_parameter_count(self):
        cfg = SimplexityModelConfig(d_vocab=2)
        model = SimplexityTransformer(cfg)
        params = count_parameters(model)
        self.assertEqual(params["total"], 142_946)
        self.assertEqual(params["trainable"], 142_946)

    def test_forward_pass_and_cache(self):
        cfg = SimplexityModelConfig(d_vocab=3, n_ctx=10)
        model = SimplexityTransformer(cfg)
        dummy_tokens = torch.randint(0, 3, (4, 10))

        logits, cache = model(dummy_tokens, return_cache=True)
        self.assertEqual(logits.shape, (4, 10, 3))
        self.assertIsNotNone(cache)
        self.assertIn("blocks.0.hook_resid_post", cache)
        self.assertIn("blocks.1.hook_resid_post", cache)
        self.assertIn("blocks.2.hook_resid_post", cache)
        self.assertIn("blocks.3.hook_resid_post", cache)
        self.assertIn("ln_final.hook_normalized", cache)
        self.assertEqual(cache["blocks.3.hook_resid_post"].shape, (4, 10, 64))

    def test_hooked_transformer_instantiation(self):
        try:
            from simplexity.model import build_hooked_transformer
            cfg = SimplexityModelConfig(d_vocab=3, n_ctx=10)
            model = build_hooked_transformer(cfg, device="cpu")
            params = count_parameters(model)
            self.assertEqual(params["total"], 143_075)

            tokens = torch.randint(0, 3, (2, 10))
            logits, cache = model.run_with_cache(tokens)
            self.assertEqual(logits.shape, (2, 10, 3))
            self.assertIn("blocks.3.hook_resid_post", cache)
            self.assertIn("ln_final.hook_normalized", cache)
        except ImportError:
            self.skipTest("transformer_lens not installed")


if __name__ == "__main__":
    unittest.main()
