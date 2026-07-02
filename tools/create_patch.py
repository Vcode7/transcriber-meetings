"""
create_patch.py -- Developer-side Incremental Patch Builder
===========================================================

Compares two Application/ directory snapshots (old release vs new release),
detects files that were ADDED or MODIFIED (by SHA-256 hash), and packages
only those files into an offline patch ZIP.

Usage
-----
    python tools/create_patch.py \
        --old  Application_v1.0/  \
        --new  Application_v1.1/  \
        --out  patch_v1.0_to_v1.1.zip \
        [--version-from 1.0] \
        [--version-to   1.1]

Output ZIP structure
--------------------
    patch_manifest.json      <- list of changed files + hashes
    files/
        backend/main.py      <- example changed file (relative path preserved)
        backend/...

The client-side updater.exe reads this ZIP and applies only the listed files.

Protected paths (never included in a patch, even if changed):
    runtime/data/            <- database
    runtime/uploads/         <- user recordings
    runtime/models/          <- encrypted AI model .dat files
    runtime/nlp-engine/      <- Qwen3 plain model directory
    runtime/backup_*/        <- previous backup dirs
    backend/.env             <- per-machine configuration
    logs/                    <- log files
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# -- Protected path prefixes (relative to the Application root) --------------
# Files under these paths are NEVER included in a patch -- they are
# machine-specific or contain user data that must be preserved.
PROTECTED_PREFIXES: List[str] = [
    "runtime/data/",
    "runtime/uploads/",
    "runtime/models/",
    "runtime/nlp-engine/",
    "runtime/backup_",
    "backend/.env",
    "logs/",
    "runtime/_hf_home/",
]


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_protected(rel_path: str) -> bool:
    """Return True if this relative path should never be included in a patch."""
    norm = rel_path.replace("\\", "/").lstrip("/")
    for prefix in PROTECTED_PREFIXES:
        if norm == prefix.rstrip("/") or norm.startswith(prefix):
            return True
    return False


def _walk_tree(root: Path) -> Dict[str, str]:
    """Walk a directory tree and return {relative_path: sha256} for every file."""
    result: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            abs_path = Path(dirpath) / fname
            rel = abs_path.relative_to(root).as_posix()
            if _is_protected(rel):
                continue
            result[rel] = _sha256(abs_path)
    return result


def build_patch(
    old_dir: Path,
    new_dir: Path,
    out_zip: Path,
    version_from: str = "unknown",
    version_to: str = "unknown",
) -> int:
    """Build a patch ZIP from old_dir to new_dir. Returns number of files included."""
    print(f"[create_patch] Scanning OLD: {old_dir}")
    old_tree = _walk_tree(old_dir)
    print(f"[create_patch] Scanning NEW: {new_dir}")
    new_tree = _walk_tree(new_dir)

    changed: List[Dict] = []

    for rel, new_hash in sorted(new_tree.items()):
        old_hash = old_tree.get(rel)
        if old_hash is None:
            status = "added"
        elif old_hash != new_hash:
            status = "modified"
        else:
            continue

        abs_new = new_dir / rel
        size = abs_new.stat().st_size
        changed.append({"path": rel, "status": status, "sha256": new_hash, "size_bytes": size})

    if not changed:
        print("[create_patch] No changes detected. No patch file written.")
        return 0

    print(f"[create_patch] {len(changed)} file(s) changed:")
    for item in changed:
        print(f"  [{item['status']:8s}] {item['path']}  ({item['size_bytes']:,} bytes)")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version_from": version_from,
        "version_to": version_to,
        "file_count": len(changed),
        "files": changed,
    }

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("patch_manifest.json", json.dumps(manifest, indent=2))
        for item in changed:
            abs_src = new_dir / item["path"]
            arc_name = f"files/{item['path']}"
            zf.write(abs_src, arc_name)

    zip_size_mb = out_zip.stat().st_size / (1024 * 1024)
    added = sum(1 for c in changed if c["status"] == "added")
    modified = sum(1 for c in changed if c["status"] == "modified")
    print(f"\n[create_patch] Patch written: {out_zip}  ({zip_size_mb:.2f} MB)")
    print(f"[create_patch] Contains {len(changed)} file(s): {added} added, {modified} modified")
    return len(changed)


def main():
    parser = argparse.ArgumentParser(
        description="Create an incremental offline patch ZIP from two Application/ snapshots."
    )
    parser.add_argument("--old", required=True, metavar="DIR",
                        help="Path to the OLD (current client) Application/ directory")
    parser.add_argument("--new", required=True, metavar="DIR",
                        help="Path to the NEW (updated) Application/ directory")
    parser.add_argument("--out", required=True, metavar="ZIP",
                        help="Output patch ZIP path (e.g. patch_v1.0_to_v1.1.zip)")
    parser.add_argument("--version-from", default="unknown", metavar="VER",
                        help="Version label for the old release")
    parser.add_argument("--version-to", default="unknown", metavar="VER",
                        help="Version label for the new release")

    args = parser.parse_args()
    old_dir = Path(args.old).resolve()
    new_dir = Path(args.new).resolve()
    out_zip = Path(args.out).resolve()

    if not old_dir.is_dir():
        print(f"ERROR: --old directory not found: {old_dir}", file=sys.stderr)
        sys.exit(1)
    if not new_dir.is_dir():
        print(f"ERROR: --new directory not found: {new_dir}", file=sys.stderr)
        sys.exit(1)

    count = build_patch(old_dir, new_dir, out_zip, args.version_from, args.version_to)
    sys.exit(0 if count >= 0 else 1)


if __name__ == "__main__":
    main()
