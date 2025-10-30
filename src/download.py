import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
import zipfile

DATASET_SLUG = "rexhaif/emojifydata-en"
DATA_FILENAME = "emojitweets-01-04-2018.txt"
RAW_DIR = Path("data") / "raw"

FILE_SHA256 = "b8ad375fcc19b45fbe2eb7cb5432d0fb6505b74b87b505bee4e5dc8049db8c5c"

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def check_config(kaggle_path: str) -> None:
    run([kaggle_path, "config", "view"]) 


def check_access(kaggle_path: str) -> None:
    run([kaggle_path, "datasets", "files", DATASET_SLUG])


def download(kaggle_path: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    compressed_file_name = f"{DATA_FILENAME}.zip"
    zip_path = raw_dir / compressed_file_name
    if zip_path.exists():
        zip_path.unlink()
    run(
        [
            kaggle_path,
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-f",
            DATA_FILENAME,
            "-p",
            str(raw_dir),
        ]
    )
    # for some reason kaggle downloads the zip with the original filename
    # change the filename from DATA_FILENAME to compressed_file_name
    original_zip_path = raw_dir / DATA_FILENAME
    if original_zip_path.exists():
        original_zip_path.rename(zip_path)

    if not zip_path.exists():
        print(
            f"Expected archive not found: {zip_path}.",
            file=sys.stderr,
        )
        raise FileNotFoundError(str(zip_path))
    return zip_path


def extract(zip_path: Path, raw_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    target = raw_dir / DATA_FILENAME
    if not target.exists():
        print(
            f"Extracted file not found: {target}.",
            file=sys.stderr,
        )
        raise FileNotFoundError(str(target))
    try:
        zip_path.unlink()
    except OSError:
        pass
    return target


def main() -> int:
    kaggle_path = shutil.which("kaggle")
    if not kaggle_path:
        print(
            "`kaggle` not found. Make sure you ran `uv sync` and you are running this script via `uv run`",
            file=sys.stderr,
        )
        return 2

    try:
        existing = RAW_DIR / DATA_FILENAME
        if existing.exists():
            try:
                digest = sha256_file(existing)
            except OSError:
                digest = None
            if digest == FILE_SHA256:
                print(f"Already present: {existing}")
                return 0
            print(f"Checksum mismatch for {existing}. Re-downloading...")
            try:
                existing.unlink()
            except OSError:
                pass
        try:
            check_config(kaggle_path)
        except subprocess.CalledProcessError:
            print(
                "Kaggle config check failed. Ensure ~/.kaggle/kaggle.json exists",
                file=sys.stderr,
            )
            return 1
        try:
            check_access(kaggle_path)
        except subprocess.CalledProcessError:
            print(
                "Kaggle dataset access check failed. Accept the dataset terms and ensure "
                "your credentials are valid: https://www.kaggle.com/datasets/rexhaif/emojifydata-en",
                file=sys.stderr,
            )
            return 1
        zip_path = download(kaggle_path, RAW_DIR)
        target = extract(zip_path, RAW_DIR)
        print(f"Downloaded to: {target}")
        return 0
    except subprocess.CalledProcessError:
        print(
            "Kaggle download failed. Check your credentials and network, "
            "and ensure dataset terms are accepted: https://www.kaggle.com/datasets/rexhaif/emojifydata-en",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
