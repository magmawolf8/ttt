"""Use official OpenPI model, checkpoint, and LIBERO transforms without training imports."""
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("OPENPI_DATA_HOME",str(ROOT/"models/cache"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")


def load_policy(checkpoint=None, backend="cuda"):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs
    from openpi.policies.policy import Policy
    from openpi.shared import normalize
    from openpi import transforms

    checkpoint = Path(checkpoint or ROOT/"models/pi05_libero").resolve()
    # Exact pi05_libero architecture at pinned OpenPI revision.
    config = Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,pytorch_compile_mode=None)
    if backend=="cuda":
        from pi05_torch import load_cuda_model
        model=load_cuda_model()
    elif backend=="cpu":
        print(f"OpenPI JAX devices: {jax.devices()}",flush=True)
        print("Loading official checkpoint in bfloat16...",flush=True)
        started=time.monotonic()
        params = model_module.restore_params(checkpoint/"params",dtype=jnp.bfloat16)
        print(f"Restored checkpoint in {time.monotonic()-started:.1f}s",flush=True)
        model=config.load(params)
        del params
    else:
        raise ValueError(f"Unknown backend {backend}")
    stats=normalize.load(checkpoint/"assets/physical-intelligence/libero")
    tokenizer=PaligemmaTokenizer(config.max_token_len)
    return Policy(model,transforms=[LiberoInputs(model_type=config.model_type),transforms.Normalize(stats,use_quantiles=True),transforms.ResizeImages(224,224),transforms.TokenizePrompt(tokenizer,discrete_state_input=False),transforms.PadStatesAndActions(32)],output_transforms=[transforms.Unnormalize(stats,use_quantiles=True),LiberoOutputs()],sample_kwargs={"num_steps":10},is_pytorch=backend=="cuda",pytorch_device="cuda" if backend=="cuda" else "cpu")
