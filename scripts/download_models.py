"""Fetch the base checkpoints the training configs point at.

ModelScope is tried first because this box reaches it at ~5 MB/s against
~1 MB/s for hf-mirror, and both are reachable while huggingface.co is not. The
Hugging Face mirror is kept as a fallback so a network change does not strand
the setup.

Downloads resume, so an interrupted run is safe to repeat. Weights land in the
repository's ``models/`` directory, which is gitignored.

    python scripts/download_models.py                        # the configured default set
    python scripts/download_models.py Qwen/Qwen3-1.7B        # one repository
    python scripts/download_models.py --source hf Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"

# 0.6B shakes out the pipeline end to end, 1.7B is the default student, and 4B
# is what a large-memory card can fine-tune without an adapter.
DEFAULT_REPOS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B")

FILE_LIST = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
FILE_RAW = "https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"
KEEP_SUFFIXES = (".json", ".txt", ".model")


def wanted(path: str) -> bool:
    """Skip documentation, images and the duplicate GGUF/original weight copies."""
    if path.startswith("."):
        return False
    return path.endswith(KEEP_SUFFIXES) or path.endswith(".safetensors")


def list_modelscope_files(repo: str) -> list[dict]:
    request = urllib.request.Request(FILE_LIST.format(repo=repo), headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode())
    if payload.get("Code") != 200:
        raise RuntimeError(f"ModelScope returned code {payload.get('Code')} for {repo}")
    return [item for item in payload["Data"]["Files"] if wanted(item["Path"])]


def fetch(url: str, destination: Path) -> None:
    """One resumable curl; --retry covers the mirror's occasional drops."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "curl", "-sSL", "--fail", "-C", "-",
            "--retry", "5", "--retry-delay", "3", "--connect-timeout", "30",
            "-o", str(destination), url,
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl exited {result.returncode} for {destination.name}")


def from_modelscope(repo: str, target: Path) -> None:
    files = list_modelscope_files(repo)
    total = sum(item["Size"] for item in files)
    print(f"  {len(files)} files, {total / 1e9:.2f} GB")
    for index, item in enumerate(files, start=1):
        destination = target / item["Path"]
        if destination.exists() and destination.stat().st_size == item["Size"]:
            print(f"  [{index}/{len(files)}] cached   {item['Path']}")
            continue
        print(f"  [{index}/{len(files)}] fetch    {item['Path']} ({item['Size'] / 1e6:.0f} MB)")
        fetch(FILE_RAW.format(repo=repo, path=item["Path"]), destination)


def from_huggingface(repo: str, target: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        local_dir=str(target),
        max_workers=4,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
    )


def ready(target: Path) -> bool:
    """A checkpoint is usable once its config and at least one weight file exist."""
    return (target / "config.json").exists() and any(target.glob("*.safetensors"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("repos", nargs="*", default=list(DEFAULT_REPOS))
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--source", choices=["auto", "modelscope", "hf"], default="auto")
    parser.add_argument("--force", action="store_true", help="re-fetch even if complete")
    args = parser.parse_args(argv)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for repo in args.repos:
        target = args.models_dir / repo.split("/")[-1]
        if ready(target) and not args.force:
            print(f"cached   {repo} -> {target}")
            continue

        print(f"fetch    {repo} -> {target}")
        for source in (["modelscope", "hf"] if args.source == "auto" else [args.source]):
            try:
                (from_modelscope if source == "modelscope" else from_huggingface)(repo, target)
                break
            except (RuntimeError, OSError, urllib.error.URLError) as error:
                print(f"  {source} failed: {error}", file=sys.stderr)
        else:
            failures += 1
            continue
        if not ready(target):
            print(f"  {target} is missing config.json or weights", file=sys.stderr)
            failures += 1

    print("\ncheckpoints:")
    for path in sorted(args.models_dir.iterdir()):
        if path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            print(f"  {path.name:24} {size / 1e9:.2f} GB")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
