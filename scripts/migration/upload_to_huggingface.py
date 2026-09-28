#!/usr/bin/env python3
"""Publish repository data files to the companion Hugging Face dataset.

Destinations come from :func:`shared.data_access.hf_path`, so the mapping used
to upload is the same one the analysis code uses to read. Files already present
on the hub with a matching size are skipped, which makes the script resumable
and safe to re-run.

Usage::

    python scripts/migration/upload_to_huggingface.py --dry-run
    python scripts/migration/upload_to_huggingface.py --include kv21/dataDistances
    python scripts/migration/upload_to_huggingface.py --all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared import data_access  # noqa: E402


def tracked_data_files() -> list[Path]:
    """Every Git-LFS tracked path, relative to the repository root."""
    out = subprocess.run(
        ["git", "lfs", "ls-files", "-n"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [Path(line) for line in out.splitlines() if line.strip()]


def plan(include: list[str] | None) -> list[tuple[Path, str, int]]:
    """Return (relative path, destination, size) for everything to publish."""
    rows = []
    for rel in tracked_data_files():
        if data_access.is_kept_in_repo(rel):
            continue
        if include and not any(str(rel).startswith(prefix) for prefix in include):
            continue
        local = REPO_ROOT / rel
        if not local.is_file():
            continue
        rows.append((rel, data_access.hf_path(rel), local.stat().st_size))
    return sorted(rows, key=lambda row: str(row[0]))


def existing_sizes(api, repo_id: str, wanted: list[str]) -> dict[str, int]:
    """Path -> size for the paths we care about.

    Queried in batches rather than by listing the whole repository, which holds
    over 100,000 structure files and is far too slow to enumerate.
    """
    sizes: dict[str, int] = {}
    for start in range(0, len(wanted), 200):
        batch = wanted[start:start + 200]
        for item in api.get_paths_info(repo_id, batch, repo_type="dataset"):
            size = getattr(item, "size", None)
            lfs = getattr(item, "lfs", None)
            if lfs is not None and getattr(lfs, "size", None):
                size = lfs.size
            if size is not None:
                sizes[item.path] = size
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include", action="append", default=None,
                        help="Only publish paths starting with this prefix. Repeatable.")
    parser.add_argument("--all", action="store_true", help="Publish everything not kept in Git.")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan and stop.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N files.")
    args = parser.parse_args()

    if not args.all and not args.include and not args.dry_run:
        parser.error("pass --all, --include PREFIX, or --dry-run")

    rows = plan(args.include)
    total = sum(size for _, _, size in rows)
    print(f"{len(rows)} files, {total / 2**30:.2f} GB, destination {data_access.HF_REPO}")

    if args.dry_run:
        for rel, dest, size in rows[:20]:
            print(f"  {size / 2**20:8.1f} MB  {rel}\n      -> {dest}")
        if len(rows) > 20:
            print(f"  ... and {len(rows) - 20} more")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    print("checking what is already published ...", flush=True)
    published = existing_sizes(api, data_access.HF_REPO, [dest for _, dest, _ in rows])

    todo = [row for row in rows if published.get(row[1]) != row[2]]
    skipped = len(rows) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    pending = sum(size for _, _, size in todo)
    print(f"  {skipped} already present, {len(todo)} to upload "
          f"({pending / 2**30:.2f} GB)\n", flush=True)

    # Commit in batches. One commit per file would leave hundreds of commits on
    # the dataset and spend most of the time on commit overhead.
    BATCH_BYTES, BATCH_FILES = 900 * 2**20, 40
    batches, current, current_bytes = [], [], 0
    for row in todo:
        if current and (current_bytes + row[2] > BATCH_BYTES or len(current) >= BATCH_FILES):
            batches.append(current)
            current, current_bytes = [], 0
        current.append(row)
        current_bytes += row[2]
    if current:
        batches.append(current)

    done_bytes, started = 0, time.time()
    for number, batch in enumerate(batches, 1):
        batch_bytes = sum(size for _, _, size in batch)
        t0 = time.time()
        api.create_commit(
            repo_id=data_access.HF_REPO,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(REPO_ROOT / rel))
                for rel, dest, _ in batch
            ],
            commit_message=f"Publish derived tables ({number}/{len(batches)})",
        )
        done_bytes += batch_bytes
        rate = batch_bytes / max(time.time() - t0, 1e-6) / 2**20
        overall = done_bytes / max(time.time() - started, 1e-6) / 2**20
        eta = (pending - done_bytes) / 2**20 / max(overall, 1e-6) / 60
        print(f"[batch {number}/{len(batches)}] {len(batch):3d} files, "
              f"{batch_bytes / 2**20:7.1f} MB in {time.time() - t0:5.1f}s "
              f"({rate:5.1f} MB/s)  eta {eta:5.1f} min", flush=True)

    print(f"\nuploaded {done_bytes / 2**30:.2f} GB in "
          f"{(time.time() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
