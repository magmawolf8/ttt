#!/bin/bash
# Native Apple Silicon setup. Pass an installed Python 3.12 executable if needed.
set -euo pipefail
cd "$(dirname "$0")"
task_python="${1:-python3}"
"$task_python" -c 'import platform,sys; assert platform.system()=="Darwin" and platform.machine()=="arm64", "Apple Silicon required"; assert sys.version_info[:2]==(3,12), "Pass a Python 3.12 executable as the first argument"'
"$task_python" -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements-mlx.txt

checkout_dependency() {
    local url="$1" destination="$2" revision="$3"
    if [ ! -d "$destination/.git" ]; then
        git clone --depth 1 "$url" "$destination"
        git -C "$destination" fetch --depth 1 origin "$revision"
        git -C "$destination" checkout --detach "$revision"
    fi
    if [ "$(git -C "$destination" rev-parse HEAD)" != "$revision" ]; then
        echo "Unexpected revision in $destination; expected $revision" >&2
        exit 1
    fi
}
checkout_dependency https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git vendor/ur_description d4042c1f5123c86645cc5dbd650b4f461f697ee9
checkout_dependency https://github.com/Physical-Intelligence/openpi.git vendor/openpi 215abfb217dbac7d5f1273282331b9b1866c0479
.venv-mlx/bin/python -m pip install --no-deps -e vendor/openpi/packages/openpi-client
.venv-mlx/bin/python -u tools/download_checkpoint.py
if [ ! -f models/pi05_libero_mlx/manifest.json ]; then
    .venv-mlx/bin/python -u tools/convert_pi05_mlx.py
fi
echo 'Ready: .venv-mlx/bin/mjpython run_policy.py --backend mlx --viewer --seconds 20 --seed 42 --output-dir runs/my_mlx_episode'
