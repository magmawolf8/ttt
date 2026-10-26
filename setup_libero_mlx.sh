#!/bin/bash
# Adds the checkpoint's native Panda environment; does not replace the card sim.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv-mlx/bin/python ] || [ ! -f models/pi05_libero_mlx/manifest.json ]; then
    echo 'First run: bash setup_mlx.sh /path/to/python3.12' >&2
    exit 1
fi
task_revision=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
if [ ! -d vendor/libero/.git ]; then
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git vendor/libero
    git -C vendor/libero fetch --depth 1 origin "$task_revision"
    git -C vendor/libero checkout --detach "$task_revision"
fi
if [ "$(git -C vendor/libero rev-parse HEAD)" != "$task_revision" ]; then
    echo "Unexpected LIBERO revision; expected $task_revision" >&2
    exit 1
fi
.venv-mlx/bin/python -m pip install -r requirements-libero-mlx.txt
.venv-mlx/bin/python -m pip install --no-deps -e vendor/libero
.venv-mlx/bin/python -c 'from libero_runtime import configure_paths; configure_paths()'
.venv-mlx/bin/python -m pip check
echo 'Ready: .venv-mlx/bin/mjpython run_libero_mlx.py --viewer'
