"""Prepare .venv-cuda on Linux/Windows using Python 3.12, without MLX."""
import argparse
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ("openpi", "https://github.com/Physical-Intelligence/openpi.git", "215abfb217dbac7d5f1273282331b9b1866c0479"),
    ("ur_description", "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git", "d4042c1f5123c86645cc5dbd650b4f461f697ee9"),
    ("libero", "https://github.com/Lifelong-Robot-Learning/LIBERO.git", "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-wheel", choices=["cu126", "cu128"], default="cu128",
                        help="Official PyTorch 2.7.1 CUDA wheel channel (default: cu128)")
    parser.add_argument("--skip-weights", action="store_true", help="Prepare dependencies only; skip checkpoint download/conversion")
    parser.add_argument("--dry-run", action="store_true", help="Print the setup commands without changing anything")
    args = parser.parse_args(argv)
    if not args.dry_run and (platform.system() not in ("Linux", "Windows") or sys.version_info[:2] != (3, 12)):
        parser.error("Run this script with Python 3.12 on Linux or Windows; use setup_mlx.sh on Apple Silicon")
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "JAX_PLATFORMS": "cpu",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OPENPI_DATA_HOME": str(ROOT / "models/cache")}
    destination = ROOT / ".venv-cuda"
    python = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def run(*command):
        command = [str(part) for part in command]
        print(subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)

    if args.dry_run:
        run(sys.executable, "-m", "venv", destination)
    else:
        venv.EnvBuilder(with_pip=True).create(destination)
    run(python, "-c", "import sys; assert sys.version_info[:2] == (3, 12), 'Use a Python 3.12 environment'")
    for name, url, revision in SOURCES:
        path = ROOT / "vendor" / name
        if not (path / ".git").exists():
            run("git", "clone", "--depth", "1", url, path)
            run("git", "-C", path, "fetch", "--depth", "1", "origin", revision)
            run("git", "-C", path, "checkout", "--detach", revision)
        if not args.dry_run:
            actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            if actual != revision:
                raise RuntimeError(f"Unexpected revision in {path}; expected {revision}, found {actual}. Existing checkout preserved.")
    run(python, "-m", "pip", "install", "torch==2.7.1", "torchvision==0.22.1",
        "--index-url", f"https://download.pytorch.org/whl/{args.cuda_wheel}")
    run(python, "-m", "pip", "install", "-r", "requirements-panda-cuda.txt")
    run(python, "-m", "pip", "install", "--no-deps", "-e", "vendor/openpi/packages/openpi-client")
    run(python, "tools/patch_transformers.py")
    run(python, "-m", "pip", "check")
    # Fail before downloading large weights if the wheel / driver / GPU is unsuitable.
    run(python, "-c", "from pi05_torch import require_cuda; require_cuda(); import torch; print(torch.cuda.get_device_name())")
    if not args.skip_weights:
        run(python, "-u", "tools/download_checkpoint.py")
        if not (ROOT / "models/pi05_libero_torch/model.safetensors.index.json").exists():
            run(python, "-u", "tools/convert_pi05_torch.py")
        run(python, "-c", "import sys; sys.path.insert(0, 'vendor/openpi/src'); from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer(200)")
    print(f"{'Planned' if args.dry_run else 'Ready'}: {python} run_panda_card.py --backend cuda --tool suction --seeds 44 --policy-noise-seed 42", flush=True)


if __name__ == "__main__":
    main()
