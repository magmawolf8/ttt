"""CPU contract tests. These do not claim to execute a CUDA model."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from panda_policy import SeededPolicy, load_panda_policy


class RecordingPolicy:
    def infer(self, observation, *, noise):
        self.observation = observation
        self.noise = noise.copy()
        self.actions = noise[0, :, :7].copy()
        return {"actions": self.actions}


class PandaPolicyTests(unittest.TestCase):
    def test_sampling_reset_and_direct_action_passthrough(self):
        native = RecordingPolicy()
        policy = SeededPolicy(native, {}, 42)
        obs = {"prompt": "Pick up the face-down card."}
        result = policy.infer(obs)
        np.testing.assert_allclose(policy.last_noise[0, 0, :3], [.30471708, -1.0399841, .7504512])
        first = policy.last_noise.copy()
        self.assertIs(native.observation, obs)
        self.assertIs(result["actions"], native.actions)
        policy.infer(obs)
        self.assertFalse(np.array_equal(policy.last_noise, first))
        policy.reset_seed(42)
        policy.infer(obs)
        np.testing.assert_array_equal(policy.last_noise, first)

    def test_supplied_noise_is_saved_without_advancing_rng(self):
        native = RecordingPolicy()
        policy = SeededPolicy(native, {}, 42)
        supplied = np.ones((1, 10, 32), dtype=np.float32)
        policy.infer({}, noise=supplied)
        np.testing.assert_array_equal(native.noise, supplied)
        supplied[:] = 0
        self.assertTrue(np.all(policy.last_noise == 1))
        policy.infer({})
        self.assertAlmostEqual(float(policy.last_noise[0, 0, 0]), .30471708)

    def test_invalid_noise_and_actions_never_reach_controller(self):
        native = RecordingPolicy()
        policy = SeededPolicy(native, {}, 42)
        for bad in [np.zeros((10, 32)), np.full((1, 10, 32), np.nan)]:
            with self.assertRaises(ValueError):
                policy.infer({}, noise=bad)
        for actions in [np.zeros((10, 32)), np.full((10, 7), np.inf)]:
            with patch.object(native, "infer", return_value={"actions": actions}):
                with self.assertRaises(ValueError):
                    policy.infer({})

    def test_cuda_dispatch_and_manifest_without_mlx(self):
        import torch
        native = RecordingPolicy()
        module = types.ModuleType("pi05_policy")
        selected = []
        def loader(*, backend):
            selected.append(backend)
            return native
        module.load_policy = loader
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models/pi05_libero_torch/model.safetensors.index.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"source": "test-only"}))
            with patch("panda_policy.ROOT", Path(directory)), patch.dict(sys.modules, {"pi05_policy": module, "pi05_mlx": None}), \
                    patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(torch.cuda, "is_bf16_supported", return_value=True), \
                    patch.object(torch.cuda, "current_device", return_value=0), \
                    patch.object(torch.cuda, "get_device_name", return_value="test GPU"), \
                    patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)):
                policy = load_panda_policy("cuda", seed=42)
                policy.infer({})
                self.assertEqual(selected, ["cuda"])
                self.assertEqual(policy.metadata["engine"], "pytorch-cuda")
                self.assertEqual(len(policy.metadata["checkpoint_manifest_sha256"]), 64)
                np.testing.assert_array_equal(native.noise, policy.last_noise)

    def test_no_cuda_fails_clearly_without_importing_mlx(self):
        code = """
import sys
from unittest.mock import patch
from panda_policy import load_panda_policy
with patch('torch.cuda.is_available', return_value=False):
    try:
        load_panda_policy('cuda')
    except RuntimeError as error:
        assert 'CUDA is unavailable' in str(error), str(error)
    else:
        raise AssertionError('Silently fell back from CUDA')
assert not any(k == 'mlx' or k.startswith('mlx.') or k == 'pi05_mlx' for k in sys.modules)
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_unsupported_gpu_rejected_before_loading_weights(self):
        from pi05_torch import require_cuda
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "bfloat16"):
                require_cuda()

    def test_custom_mlx_bundle_is_actually_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "manifest.json").write_text('{"test":true}')
            native = RecordingPolicy()
            native.metadata = {"checkpoint_path":str(checkpoint), "engine":"mlx", "finetuning":{"selected_step":20}}
            module = types.ModuleType("pi05_mlx")
            calls = []
            def factory(*, checkpoint, seed):
                calls.append((checkpoint, seed))
                return native
            module.MLXPolicy = factory
            with patch.dict(sys.modules, {"pi05_mlx":module}):
                policy = load_panda_policy("mlx", seed=5, checkpoint=checkpoint)
            self.assertEqual(calls, [(checkpoint, 5)])
            self.assertEqual(policy.metadata["finetuning"]["selected_step"], 20)
            self.assertEqual(len(policy.metadata["checkpoint_manifest_sha256"]), 64)

    def test_mlx_checkpoint_cannot_silently_load_original_cuda_weights(self):
        with self.assertRaisesRegex(ValueError, "matching PyTorch export"):
            load_panda_policy("cuda", checkpoint="runs/trained/checkpoint")


if __name__ == "__main__":
    unittest.main()
