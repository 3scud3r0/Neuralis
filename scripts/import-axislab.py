#!/usr/bin/env python3
"""Validate and reconstruct the user-supplied AxisLab consolidated text bundle.

Read payloads by declared byte count, never by searching for an END marker inside
source text. Validate the entire archive before publishing a new destination.
Existing destinations are deliberately refused; import into a new directory to
compare or refresh provenance without overwriting local work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

ROOT = "axislab-research-0.3.0"
COUNT = 28
BEGIN = re.compile(rb"===== BEGIN FILE: ([^\r\n]+) =====\n")
META = re.compile(rb"# SHA256: ([0-9a-f]{64})\n# BYTE_LENGTH: (0|[1-9][0-9]*)\n")


class BundleError(ValueError):
    """Malformed, unsafe, incomplete, or corrupt bundle."""


def safe_relative(raw: bytes) -> str:
    try:
        name = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BundleError("Path is not UTF-8") from error
    parts = name.split("/")
    if (len(parts) < 2 or parts[0] != ROOT or any(
        not part or part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
        for part in parts
    ) or PurePosixPath(name).is_absolute()):
        raise BundleError(f"Unsafe or malformed archive path: {name!r}")
    return "/".join(parts[1:])


def parse_bundle(data: bytes) -> list[dict]:
    first = BEGIN.search(data)
    if first is None:
        raise BundleError("Missing BEGIN FILE marker")
    preamble = data[:first.start()]
    if b"===== BEGIN" in preamble or b"===== END" in preamble:
        raise BundleError("Malformed marker in preamble")
    declared = re.search(rb"^Arquivos textuais: ([0-9]+)$", preamble, re.MULTILINE)
    if declared is None or int(declared[1]) != COUNT:
        raise BundleError(f"Expected preamble declaring {COUNT} text files")
    entries, seen, position = [], set(), first.start()
    while position < len(data):
        begin = BEGIN.match(data, position)
        if begin is None:
            raise BundleError(f"Malformed BEGIN marker at byte {position}")
        relative = safe_relative(begin[1])
        if relative in seen:
            raise BundleError(f"Duplicate archive path: {relative}")
        seen.add(relative)
        metadata = META.match(data, begin.end())
        if metadata is None:
            raise BundleError(f"Malformed metadata for {relative}")
        length = int(metadata[2])
        start, end = metadata.end(), metadata.end() + length
        if end > len(data):
            raise BundleError(f"Truncated payload for {relative}")
        payload = data[start:end]
        digest = hashlib.sha256(payload).hexdigest()
        if digest != metadata[1].decode("ascii"):
            raise BundleError(f"SHA256 mismatch for {relative}")
        terminator = b"\n===== END FILE: " + begin[1] + b" =====\n"
        if not data.startswith(terminator, end):
            raise BundleError(f"BYTE_LENGTH or END marker mismatch for {relative}")
        position = end + len(terminator)
        while position < len(data) and data[position:position + 1] == b"\n":
            position += 1
        target = "README.original.md" if relative == "README.md" else relative
        entries.append({"original_path": ROOT + "/" + relative,
                        "path": target, "byte_length": length,
                        "sha256": digest, "payload": payload})
    if len(entries) != COUNT:
        raise BundleError(f"Expected {COUNT} files, found {len(entries)}")
    targets = {entry["path"] for entry in entries}
    if len(targets) != len(entries) or "provenance.json" in targets:
        raise BundleError("Destination path collision")
    for target in targets:
        if any(str(parent) in targets for parent in PurePosixPath(target).parents):
            raise BundleError(f"File/directory path collision: {target}")
    return entries


def import_bundle(source: Path, destination: Path, *, check: bool = False) -> dict:
    data = source.read_bytes()
    entries = parse_bundle(data)
    manifest = {
        "schema_version": 1,
        "source_bundle": source.name,
        "source_bundle_sha256": hashlib.sha256(data).hexdigest(),
        "archive_root": ROOT,
        "text_file_count": len(entries),
        "license": "MIT (original LICENSE preserved byte for byte)",
        "excluded_binary_data": "Original bundle explicitly excludes .npz files; none were supplied.",
        "readme_mapping": "README.md is preserved as README.original.md; English README.md is maintained separately.",
        "files": [{key: value for key, value in entry.items() if key != "payload"} for entry in entries],
        "source_modifications": [],
    }
    if check:
        return manifest
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise BundleError(f"Refusing to overwrite destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".axislab-import-", dir=destination.parent))
    try:
        for entry in entries:
            target = staging / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(entry["payload"])
        (staging / "provenance.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # A competing process must not create a destination while importing.
        if destination.exists() or destination.is_symlink():
            raise BundleError(f"Destination appeared during import: {destination}")
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path, nargs="?", default=Path("packages/axislab"))
    parser.add_argument("--check", action="store_true", help="Validate all entries without writing files")
    args = parser.parse_args()
    try:
        manifest = import_bundle(args.bundle, args.destination, check=args.check)
    except (BundleError, OSError) as error:
        parser.exit(1, f"AxisLab import rejected: {error}\n")
    print(f"Validated {manifest['text_file_count']} byte-exact files; bundle SHA256 {manifest['source_bundle_sha256']}")
    if not args.check:
        print(f"Imported into {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
