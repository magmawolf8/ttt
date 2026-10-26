"""Backward-compatible MLX entry point; the shared runner also supports CUDA."""
from run_panda_card import dashboard, main, sha256, verify_scene  # noqa: F401


if __name__ == "__main__":
    main(default_backend="mlx")
