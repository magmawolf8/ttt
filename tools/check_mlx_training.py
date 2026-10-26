"""Compare the MLX flow objective against the pinned upstream JAX FP32 model.

Uses explicit identical actions, noise and time. The JAX forward is the body of
upstream Pi0.compute_loss, without its internal RNG/augmentation wrapper.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor/openpi/src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=70)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from train_panda_mlx import MLXPolicy, Pi05MLX, mx, action_chunk, flow_matching_loss
    import numpy as np
    with np.load(args.episode) as episode:
        inputs = {"observation/image":episode["image"][args.frame],
                  "observation/wrist_image":episode["wrist_image"][args.frame],
                  "observation/state":episode["state"][args.frame], "prompt":str(episode["prompt"])}
        raw_actions = action_chunk(episode["actions"], args.frame)
    policy = MLXPolicy()
    images, tokens = policy.prepare(inputs)
    cache, length = policy.model.encode_prefix(images, tokens)
    normalized = (raw_actions - policy.q01) / (policy.q99 - policy.q01 + 1e-6) * 2 - 1
    actions = np.pad(normalized, ((0, 0), (0, 25)))[None].astype(np.float32)
    noise = np.random.default_rng(321).standard_normal((1, 10, 32)).astype(np.float32)
    time = np.array([.37], dtype=np.float32)
    noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    candidate = np.asarray(policy.model.denoise(mx.array(noisy), mx.array(time), cache, length))
    target = noise - actions
    candidate_loss = float(np.mean(np.square(candidate - target)))
    # Gradient check through the entire expert, not only the final output bias.
    name = "action_in_proj/kernel"
    frozen = {k:v for k,v in policy.model.weights.items() if k != name}
    parameter = policy.model.weights[name]
    def objective(weight):
        return flow_matching_loss({name:weight}, frozen, cache, length,
                                  mx.array(actions), mx.array(noise), mx.array(time))
    gradient = mx.grad(objective)(parameter)
    mx.eval(gradient)
    index = np.unravel_index(int(mx.argmax(mx.abs(gradient)).item()), gradient.shape)
    epsilon = .001
    direction = mx.zeros_like(parameter).at[index].add(epsilon)
    finite_difference = (objective(parameter + direction).item() - objective(parameter - direction).item()) / (2 * epsilon)
    analytic = float(gradient[index].item())
    gradient_error = abs(finite_difference - analytic) / max(abs(analytic), 1e-8)
    # Release the MLX model before constructing the independent reference.
    del policy, cache, frozen, parameter, gradient, images, tokens, direction
    gc.collect()
    mx.clear_cache()

    import jax
    import jax.numpy as jnp
    import sentencepiece
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.pi0 import make_attn_mask
    from openpi.models.model import Observation, restore_params
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.shared import normalize, nnx_utils
    from openpi import transforms
    config = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
                      dtype="float32", pytorch_compile_mode=None)
    model = config.load(restore_params(ROOT / "models/pi05_libero/params", dtype=jnp.float32))
    tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
    tokenizer._max_len = 200
    tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "models/pi05_libero_mlx/paligemma_tokenizer.model"))
    stats = normalize.load(ROOT / "models/pi05_libero/assets/physical-intelligence/libero")
    prepare = transforms.compose([LiberoInputs(model_type=config.model_type),
                                  transforms.Normalize(stats, use_quantiles=True),
                                  transforms.ResizeImages(224,224),
                                  transforms.TokenizePrompt(tokenizer, discrete_state_input=False),
                                  transforms.PadStatesAndActions(32)])
    prepared = prepare({**inputs, "actions":raw_actions.copy()})
    np.testing.assert_allclose(prepared["actions"], actions[0], rtol=1e-6, atol=1e-6)
    obs = Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None], prepared))
    # Attach a bound method so upstream module_jit handles NNX model state.
    import types
    def forward(self, observation, noisy_actions, timestep):
        prefix, pm, pa = self.embed_prefix(observation)
        suffix, sm, sa, cond = self.embed_suffix(observation, noisy_actions, timestep)
        mask = jnp.concatenate([pm, sm], axis=1)
        ar = jnp.concatenate([pa, sa], axis=0)
        (_, output), _ = self.PaliGemma.llm([prefix,suffix], mask=make_attn_mask(mask,ar),
                                          positions=jnp.cumsum(mask,axis=1)-1, adarms_cond=[None,cond])
        return self.action_out_proj(output[:, -self.action_horizon:])
    reference = np.asarray(nnx_utils.module_jit(types.MethodType(forward, model))(obs, jnp.asarray(noisy), jnp.asarray(time)))
    relative = float(np.linalg.norm(candidate-reference) / max(np.linalg.norm(reference),1e-12))
    reference_loss = float(np.mean(np.square(reference-target)))
    report = {"reference":"pinned OpenPI Pi0.compute_loss forward, float32, fixed time/noise; no augmentation",
              "episode":str(args.episode), "frame":args.frame, "time":time.tolist(),
              "mlx_loss":candidate_loss, "jax_loss":reference_loss,
              "velocity_relative_l2":relative, "velocity_max_abs":float(np.max(np.abs(candidate-reference))),
              "finite_difference":{"parameter":name, "index":list(map(int,index)), "epsilon":epsilon,
                                   "analytic":analytic,"numerical":finite_difference,"relative_error":gradient_error},
              "passed":relative<.001 and abs(candidate_loss-reference_loss)<.001 and gradient_error<.02}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
