"""Download the unmodified public OpenPI checkpoint with resumable HTTP reads."""
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "checkpoints/pi05_libero/"
DEST = ROOT / "models" / "pi05_libero"


def download_one(item):
    name, size = item["name"], int(item["size"])
    path = DEST / name.removeprefix(PREFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size:
        return
    partial = path.with_name(path.name + ".partial")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            "https://storage.googleapis.com/openpi-assets/" + name,
            headers={"Range": f"bytes={offset}-"} if offset else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and response.status != 206:
                    offset = 0
                with partial.open("ab" if offset else "wb") as handle:
                    while chunk := response.read(4 * 1024 * 1024):
                        handle.write(chunk)
            if partial.stat().st_size != size:
                raise IOError(f"Wrong length: {name}")
            partial.replace(path)
            print(f"Finished {name} ({size / 1e6:.1f} MB)", flush=True)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2)


def main():
    url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?" + urllib.parse.urlencode(
        {"prefix": PREFIX, "maxResults": 500}
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        listing = json.load(response)
    if "nextPageToken" in listing:
        raise RuntimeError("Checkpoint listing unexpectedly requires pagination")
    items = listing["items"]
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "download_manifest.json").write_text(json.dumps(listing, indent=2))
    total = sum(int(item["size"]) for item in items)
    print(f"Downloading official pi05_libero: {total / 1e9:.2f} GB", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(download_one, item) for item in items]
        while not all(f.done() for f in futures):
            done = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
            print(f"Download progress: {done / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
            concurrent.futures.wait(futures, timeout=30)
        for future in futures:
            future.result()
    print("Checkpoint download complete", flush=True)


if __name__ == "__main__":
    main()
