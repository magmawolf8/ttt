"""Apply the pinned upstream patches required by OpenPI's PyTorch backend."""
from pathlib import Path
import shutil
import transformers

ROOT=Path(__file__).resolve().parents[1]


def main():
    if transformers.__version__!="4.53.2":
        raise RuntimeError("OpenPI requires transformers==4.53.2 for these patches")
    patches=ROOT/"vendor/openpi/src/openpi/models_pytorch/transformers_replace"
    installed=Path(transformers.__file__).resolve().parent
    backups=ROOT/"models/transformers-original"
    for source in patches.rglob("*.py"):
        relative=source.relative_to(patches)
        target=installed/relative
        backup=backups/relative
        if target.exists() and not backup.exists():
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,backup)
        shutil.copy2(source,target)
        print(relative)


if __name__=="__main__":
    main()
