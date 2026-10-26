# OpenPI model equations

The MLX implementation in `pi05_mlx.py`, the flow-matching objective in
`train_panda_mlx.py`, and its reference check in `tools/check_mlx_training.py`
follow the model equations and checkpoint
layout in Physical Intelligence's OpenPI at revision
`215abfb217dbac7d5f1273282331b9b1866c0479`:
https://github.com/Physical-Intelligence/openpi

OpenPI is licensed under the Apache License, Version 2.0. Its Gemma and SigLIP
implementations include copyright notices for the Big Vision Authors (2024 and
2023 respectively). The upstream license is retained with the pinned source in
`vendor/openpi/LICENSE`; a permanent copy is in `licenses/openpi-Apache-2.0.txt`.
Downloaded model weights retain their upstream license terms. They are not
committed here.

The community `papasanimohansrinivas/pi05-mlx` project was inspected during
development, but is not a runtime dependency and its code/weights are not bundled.

The native Panda runner (`run_libero_mlx.py`, `libero_runtime.py`) follows the
environment, preprocessing, and action execution conventions of the same pinned
OpenPI `examples/libero/main.py` (Apache-2.0). LIBERO is retrieved separately at
`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`; its source and assets retain their
upstream notices in `vendor/libero`. The LIBERO project is MIT-licensed.
