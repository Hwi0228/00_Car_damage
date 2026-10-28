from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath


DATA_ROOT = Path(__file__).resolve().parent / "data"
OUTPUT_ROOT = DATA_ROOT / "extracted"
ARCHIVES = (
    ("train", "images", "TS_damage.zip"),
    ("train", "annotations", "TL_damage.zip"),
    ("validation", "images", "VS_damage.zip"),
    ("validation", "annotations", "VL_damage.zip"),
)
SAFETY_RESERVE_BYTES = 5 * 1024**3


def find_archive(name: str) -> Path:
    matches = list(DATA_ROOT.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {name} under {DATA_ROOT}, found {len(matches)}."
        )
    return matches[0]


def archive_stats(path: Path) -> tuple[int, int]:
    file_count = 0
    uncompressed_bytes = 0
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            member = PurePosixPath(entry.filename)
            if (
                member.is_absolute()
                or ".." in member.parts
                or not member.parts
                or member.parts[0] != "damage"
            ):
                raise ValueError(f"Unexpected path in {path.name}: {entry.filename}")
            if entry.is_dir():
                continue
            file_type = (entry.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise ValueError(f"Symbolic link is not allowed: {entry.filename}")
            file_count += 1
            uncompressed_bytes += entry.file_size
    if not file_count:
        raise ValueError(f"No files found in {path}.")
    return file_count, uncompressed_bytes


def output_stats(path: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            count += 1
            size += file_path.stat().st_size
    return count, size


def show_progress(count: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 0.001)
    rate = count / elapsed
    percent = count * 100 / total
    width = 24
    filled = round(width * min(count / total, 1))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\r  [{bar}] {percent:5.1f}% "
        f"{count:,}/{total:,} files - {rate:,.0f} files/s"
    )
    sys.stdout.flush()


def extract(path: Path, staging_root: Path, expected_count: int) -> None:
    tar = shutil.which("tar")
    if tar:
        process = subprocess.Popen(
            [tar, "-xvf", str(path), "-C", str(staging_root)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stderr is not None
        started = time.monotonic()
        count = 0
        last_update = started
        try:
            for entry_name in process.stderr:
                if entry_name.strip() and not entry_name.rstrip().endswith("/"):
                    count += 1
                    now = time.monotonic()
                    if now - last_update >= 0.5 or count == expected_count:
                        show_progress(count, expected_count, started)
                        last_update = now
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stderr.close()
        if return_code:
            raise subprocess.CalledProcessError(return_code, process.args)
        if count != expected_count:
            show_progress(count, expected_count, started)
        sys.stdout.write("\n")
        return

    with zipfile.ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        started = time.monotonic()
        last_update = started
        for count, entry in enumerate(entries, start=1):
            archive.extract(entry, staging_root)
            now = time.monotonic()
            if now - last_update >= 0.5 or count == expected_count:
                show_progress(count, expected_count, started)
                last_update = now
    sys.stdout.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract AI Hub damage data archives sequentially into a clean folder structure."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect archive sizes and disk space without extracting.",
    )
    args = parser.parse_args()

    plans = []
    pending_bytes = 0
    for split, kind, archive_name in ARCHIVES:
        archive_path = find_archive(archive_name)
        expected_count, expected_bytes = archive_stats(archive_path)
        destination = OUTPUT_ROOT / split / kind
        already_complete = False
        if destination.exists():
            actual_count, actual_bytes = output_stats(destination)
            already_complete = (actual_count, actual_bytes) == (
                expected_count,
                expected_bytes,
            )
            if not already_complete:
                raise FileExistsError(
                    f"{destination} already exists but does not match {archive_name}; "
                    "inspect it before removing or replacing it."
                )
        if not already_complete:
            pending_bytes += expected_bytes
        plans.append(
            (
                archive_path,
                destination,
                expected_count,
                expected_bytes,
                already_complete,
            )
        )
        print(
            f"{archive_name}: {expected_count:,} files, "
            f"{expected_bytes / 1024**3:.2f} GiB uncompressed"
            + (" [already complete]" if already_complete else "")
        )

    free_bytes = shutil.disk_usage(OUTPUT_ROOT.parent).free
    required_bytes = pending_bytes + (SAFETY_RESERVE_BYTES if pending_bytes else 0)
    print(
        f"Free disk: {free_bytes / 1024**3:.2f} GiB; "
        f"needed including reserve: {required_bytes / 1024**3:.2f} GiB"
    )
    if free_bytes < required_bytes:
        raise OSError(
            "Not enough free disk space. Free space or change OUTPUT_ROOT before extracting."
        )
    if args.dry_run:
        print("Dry run complete; no files were extracted.")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    staging_root = OUTPUT_ROOT / "_staging"
    staging_root.mkdir(exist_ok=True)
    for archive_path, destination, expected_count, expected_bytes, complete in plans:
        if complete:
            continue

        stage = staging_root / f"{destination.parent.name}_{destination.name}"
        if stage.exists() and not (stage / "damage").is_dir():
            raise FileExistsError(
                f"Incomplete staging directory found at {stage}. "
                "Inspect it and remove only that directory before rerunning."
            )
        stage.mkdir(exist_ok=True)
        print(f"Extracting {archive_path.name} -> {destination}", flush=True)
        started = time.monotonic()
        extract(archive_path, stage, expected_count)

        extracted_root = stage / "damage"
        actual_count, actual_bytes = output_stats(extracted_root)
        if (actual_count, actual_bytes) != (expected_count, expected_bytes):
            raise OSError(
                f"Extraction verification failed for {archive_path.name}: "
                f"expected {expected_count} files/{expected_bytes} bytes, "
                f"got {actual_count} files/{actual_bytes} bytes. "
                f"Partial data remains at {stage}."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        extracted_root.rename(destination)
        stage.rmdir()
        print(
            f"Done {archive_path.name} in {(time.monotonic() - started) / 60:.1f} min.",
            flush=True,
        )

    try:
        staging_root.rmdir()
    except OSError:
        pass

    print(f"All archives extracted under: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
