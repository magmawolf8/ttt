"""Dataset boundary, frozen-backbone and flow-objective contract checks."""
import os
os.environ["MLX_ENABLE_TF32"] = "0"

import importlib.util
import unittest
from unittest.mock import patch
import numpy as np

if importlib.util.find_spec("mlx") is None:
    raise unittest.SkipTest("MLX training tests require Apple Silicon dependencies")
from train_panda_mlx import action_chunk, is_action_parameter, flow_matching_loss, phase_pools, sample_refs, mx


class PandaTrainingTests(unittest.TestCase):
    def test_phase_sampling_does_not_drown_contact_in_holds(self):
        episodes = {"a": {"phases": ["hold"] * 100 + ["contact", "recover_align"]}}
        pools = phase_pools(episodes, [("a", i) for i in range(102)])
        refs = sample_refs(pools, 3000, np.random.default_rng(42))
        counts = {phase:sum(ref in candidates for ref in refs) for phase,candidates in pools.items()}
        self.assertEqual(set(counts), {"hold", "contact", "recovery"})
        self.assertTrue(all(850 < value < 1150 for value in counts.values()), counts)

    def test_chunk_does_not_cross_episode_boundary(self):
        actions = np.arange(21).reshape(3, 7)
        chunk = action_chunk(actions, 1, horizon=5)
        np.testing.assert_array_equal(chunk[0], actions[1])
        np.testing.assert_array_equal(chunk[1:], np.repeat(actions[2:3], 4, axis=0))

    def test_freeze_selector_keeps_all_vision_and_prefix_weights(self):
        for name in ["PaliGemma/img/Transformer/encoderblock/LayerNorm_1/bias",
                     "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel",
                     "PaliGemma/llm/layers/attn/q_einsum/w",
                     "PaliGemma/llm/embedder/input_embedding"]:
            self.assertFalse(is_action_parameter(name), name)
        for name in ["PaliGemma/llm/layers/attn/q_einsum_1/w",
                     "PaliGemma/llm/final_norm_1/Dense_0/kernel",
                     "action_out_proj/kernel", "time_mlp_in/kernel"]:
            self.assertTrue(is_action_parameter(name), name)

    def test_flow_matching_time_and_velocity_convention(self):
        # CPU-only test of the objective, independent of the large neural model.
        with mx.stream(mx.cpu):
            actions = mx.full((2, 10, 32), 2.)
            noise = mx.full((2, 10, 32), 3.)
            timestep = mx.array([.25, .75])
            seen = {}
            class Denoiser:
                def __init__(self, weights, eager):
                    pass
                def denoise(self, noisy, time, cache, length):
                    seen["x"] = np.asarray(noisy)
                    return mx.zeros_like(noisy)
            with patch("train_panda_mlx.Pi05MLX", Denoiser):
                loss = flow_matching_loss({}, {}, [], 0, actions, noise, timestep)
                self.assertEqual(float(loss.item()), 1.)
            np.testing.assert_allclose(seen["x"][0], 2.25)
            np.testing.assert_allclose(seen["x"][1], 2.75)


if __name__ == "__main__":
    unittest.main()
