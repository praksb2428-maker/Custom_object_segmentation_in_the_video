"""Download one official Meta checkpoint, without overwriting an existing file."""

import argparse
from pathlib import Path
import urllib.request

BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
FILENAMES = {"small": "sam2.1_hiera_small.pt", "base-plus": "sam2.1_hiera_base_plus.pt",
             "large": "sam2.1_hiera_large.pt"}


def main():
    parser = argparse.ArgumentParser(description="Download SAM 2.1 from the official Meta server")
    parser.add_argument("--model", choices=FILENAMES, default="small")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "checkpoints"
    directory.mkdir(exist_ok=True)
    target = directory / FILENAMES[args.model]
    partial = target.with_suffix(".pt.part")
    if target.exists() or partial.exists():
        raise SystemExit(f"Already exists; inspect before downloading again: {target} or {partial}")
    request = urllib.request.Request(BASE + target.name, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
        expected = int(response.headers.get("Content-Length", 0))
        count = 0
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            count += len(chunk)
        if expected and count != expected:
            raise RuntimeError("Incomplete download; .part file retained")
    partial.rename(target)
    print(f"Downloaded {count} bytes: {target}")


if __name__ == "__main__":
    main()
