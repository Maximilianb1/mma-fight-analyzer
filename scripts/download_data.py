"""Download the labeled MMA clip dataset from Google Drive and unpack it to data/raw/.

The dataset is a Drive folder containing one ZIP per fight (each ZIP holds
clip_XXXX.mp4 files + <Fight>_labels.csv). Set DRIVE_FOLDER_ID below (or pass
--folder-id) to the shared folder's ID.

  python scripts/download_data.py
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

DRIVE_FOLDER_ID = "1bYKlbqa-Mu7UKUIez7EfxA7xtqWvXwyI"


def normalize_extracted(extract_dir, out_dir):
    """Whatever the ZIP layout, end with data/raw/<Fight>/{clips,csv}."""
    for csv in Path(extract_dir).rglob("*_labels.csv"):
        fight = csv.name[:-len("_labels.csv")]
        dest = Path(out_dir) / fight
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv, dest / csv.name)
        for mp4 in Path(extract_dir).rglob("clip_*.mp4"):
            shutil.copy2(mp4, dest / mp4.name)
        return fight
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder-id", default=DRIVE_FOLDER_ID)
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()

    import gdown
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        gdown.download_folder(id=args.folder_id, output=tmp, quiet=False, use_cookies=False)
        zips = sorted(Path(tmp).rglob("*.zip"))
        print(f"downloaded {len(zips)} archives")
        for z in zips:
            with tempfile.TemporaryDirectory() as ext:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(ext)
                fight = normalize_extracted(ext, out_dir)
                print(f"  {z.name} -> {fight}")
        # the Drive folder also carries fights_meta.csv; use it if the repo copy is missing
        metas = list(Path(tmp).rglob("fights_meta.csv"))
        local_meta = Path("data/fights_meta.csv")
        if metas and not local_meta.exists():
            shutil.copy2(metas[0], local_meta)
            print(f"copied fights_meta.csv -> {local_meta}")
    print(f"\nDataset ready under {out_dir}")


if __name__ == "__main__":
    main()
