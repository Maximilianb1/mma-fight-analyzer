"""Download the frozen deployment checkpoints from the project's GitHub release.

Run from the repository root:
    python scripts/download_models.py
"""

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


RELEASE_URL = (
    "https://github.com/Maximilianb1/mma-fight-analyzer/releases/download/v1.0-models"
)

MODELS = {
    Path("outputs/gate/gate.pt"): (
        "gate.pt",
        "a0735e9eb687a3c4e4924db1469891b3a26c28a481e6571bf4e1885915728f08",
    ),
    Path("outputs/phase/deployment_phase_final.pt"): (
        "deployment_phase_final.pt",
        "4c87f74bd79f0031daad357f8b89052d866ba40833563a1a2689d3ab9df38187",
    ),
    Path("outputs/phase/deployment_pressure_final.pt"): (
        "deployment_pressure_final.pt",
        "0a2b088b6709bbba10b8c5d1515e3760d440fed3cc83e1aee0313a2afc93e85a",
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination):
    """Stream one URL to a temporary file and atomically move it into place."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="replace valid existing files"
    )
    args = parser.parse_args()

    for destination, (asset, expected_hash) in MODELS.items():
        if (
            destination.is_file()
            and sha256(destination) == expected_hash
            and not args.force
        ):
            print(f"ready: {destination}")
            continue
        print(f"downloading {asset} -> {destination}")
        download(f"{RELEASE_URL}/{asset}", destination)
        actual_hash = sha256(destination)
        if actual_hash != expected_hash:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {asset}: {actual_hash}")

    print("All deployment checkpoints are ready.")


if __name__ == "__main__":
    main()
