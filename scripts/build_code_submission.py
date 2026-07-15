"""Build the code-only ZIP required for the Moodle project submission."""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT.parent / "mma-fight-analyzer-code-submission.zip"
ARCHIVE_ROOT = "mma-fight-analyzer"

ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "README.md",
    "requirements.txt",
}
CODE_DIRECTORIES = {
    ".streamlit",
    "docs",
    "notebooks",
    "scripts",
    "src",
    "tests",
    "tools",
}
FORBIDDEN_SUFFIXES = {
    ".avi",
    ".ckpt",
    ".mkv",
    ".mov",
    ".mp4",
    ".npz",
    ".pt",
    ".pth",
    ".tar",
    ".zip",
}


def tracked_files() -> list[PurePosixPath]:
    """Return repository files tracked by Git, using POSIX paths."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        PurePosixPath(item)
        for item in result.stdout.decode("utf-8").split("\0")
        if item
    ]


def should_include(path: PurePosixPath) -> bool:
    """Select code and small supporting files, excluding generated artifacts."""
    text = path.as_posix()
    if text in ROOT_FILES or text == "data/fights_meta.csv":
        return True
    if path.parts[:2] == ("report", "scripts"):
        return True
    return bool(path.parts and path.parts[0] in CODE_DIRECTORIES)


def build_archive(output: Path) -> tuple[int, int]:
    """Write the curated archive and return its file count and uncompressed size."""
    selected = sorted(path for path in tracked_files() if should_include(path))
    if not selected:
        raise RuntimeError("No tracked submission files were found.")

    for relative in selected:
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"Large artifact selected unexpectedly: {relative}")
        if not (REPOSITORY_ROOT / Path(*relative.parts)).is_file():
            raise FileNotFoundError(relative)

    required = {"README.md", "requirements.txt"}
    names = {path.as_posix() for path in selected}
    missing = required - names
    if missing:
        raise RuntimeError(f"Required files missing from archive: {sorted(missing)}")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    total_size = 0
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in selected:
            source = REPOSITORY_ROOT / Path(*relative.parts)
            archive_name = f"{ARCHIVE_ROOT}/{relative.as_posix()}"
            archive.write(source, archive_name)
            total_size += source.stat().st_size

    temporary.replace(output)
    return len(selected), total_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"ZIP destination (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    count, total_size = build_archive(args.output)
    print(f"Created: {args.output.resolve()}")
    print(f"Files: {count}; uncompressed size: {total_size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
