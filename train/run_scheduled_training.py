from __future__ import annotations

import ctypes
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TextIO
from zoneinfo import ZoneInfo

import extract_dataset


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
LOG_PATH = BASE_DIR / "scheduled_training.log"
SEOUL = ZoneInfo("Asia/Seoul")
DEADLINE = datetime(2026, 10, 29, 16, 0, tzinfo=SEOUL)
INITIAL_CHECKS = (
    datetime(2026, 10, 28, 23, 0, tzinfo=SEOUL),
    datetime(2026, 10, 29, 0, 0, tzinfo=SEOUL),
)
STABILITY_SECONDS = 60
RETRY_DELAY_SECONDS = 5 * 60
MAX_PIPELINE_ATTEMPTS = 3
SOURCE_ARCHIVES = ("TS_damage.zip", "TL_damage.zip", "VS_damage.zip", "VL_damage.zip")
PARTIAL_SUFFIXES = (".crdownload", ".part", ".tmp", ".download")


def log(message: str, stream: TextIO) -> None:
    line = f"[{datetime.now(SEOUL).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


def keep_computer_awake(stream: TextIO) -> None:
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        es_continuous = 0x80000000
        es_system_required = 0x00000001
        if not kernel32.SetThreadExecutionState(
            es_continuous | es_system_required
        ):
            log("WARNING: Windows could not be kept awake.", stream)
    except (AttributeError, OSError) as error:
        log(f"WARNING: Could not prevent system sleep: {error}", stream)


def wait_until(target: datetime, stream: TextIO) -> None:
    while True:
        remaining = (target - datetime.now(SEOUL)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 60))


def archive_paths() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name in SOURCE_ARCHIVES:
        matches = list(DATA_ROOT.rglob(name)) if DATA_ROOT.exists() else []
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {name} under {DATA_ROOT}, found {len(matches)}."
            )
        result[name] = matches[0]
    return result


def has_partial_downloads() -> list[Path]:
    if not DATA_ROOT.exists():
        return []
    return [
        path
        for path in DATA_ROOT.rglob("*")
        if path.is_file() and path.name.casefold().endswith(PARTIAL_SUFFIXES)
    ]


def archives_ready(stream: TextIO) -> bool:
    if partials := has_partial_downloads():
        log(
            "Download is not complete; partial files remain: "
            + ", ".join(str(path) for path in partials[:10]),
            stream,
        )
        return False

    try:
        paths = archive_paths()
    except FileNotFoundError as error:
        log(f"Download is not complete: {error}", stream)
        return False

    first_sizes = {name: path.stat().st_size for name, path in paths.items()}
    if any(size == 0 for size in first_sizes.values()):
        log("Download is not complete; at least one archive is empty.", stream)
        return False

    log(
        f"Found all four archives ({sum(first_sizes.values()) / 1024**3:.2f} GiB); "
        f"checking that sizes stay unchanged for {STABILITY_SECONDS} seconds.",
        stream,
    )
    time.sleep(STABILITY_SECONDS)
    try:
        paths = archive_paths()
    except FileNotFoundError as error:
        log(f"Archive list changed while checking: {error}", stream)
        return False
    second_sizes = {name: path.stat().st_size for name, path in paths.items()}
    if first_sizes != second_sizes or has_partial_downloads():
        log("Download is still changing; no files will be extracted yet.", stream)
        return False

    for name, path in paths.items():
        try:
            count, uncompressed_bytes = extract_dataset.archive_stats(path)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            log(f"{name} is not a complete readable ZIP: {error}", stream)
            return False
        log(
            f"Verified ZIP index: {name} ({count:,} files, "
            f"{uncompressed_bytes / 1024**3:.2f} GiB unpacked).",
            stream,
        )
    return True


def run_command(command: list[str], stream: TextIO, label: str) -> int:
    log(f"Starting {label}: {' '.join(command)}", stream)
    process = subprocess.Popen(
        command,
        cwd=BASE_DIR.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            stream.write(line)
            stream.flush()
        return_code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if return_code:
        log(f"{label} exited with code {return_code}.", stream)
    else:
        log(f"{label} completed successfully.", stream)
    return return_code


def run_pipeline(stream: TextIO) -> int:
    python = sys.executable
    steps = (
        (
            "dataset extraction",
            [python, "-u", str(BASE_DIR / "extract_dataset.py")],
        ),
        (
            "pair checking and training",
            [python, "-u", str(BASE_DIR / "prep.py")],
        ),
    )
    for attempt in range(1, MAX_PIPELINE_ATTEMPTS + 1):
        log(f"Pipeline attempt {attempt}/{MAX_PIPELINE_ATTEMPTS}.", stream)
        failed = False
        for label, command in steps:
            if datetime.now(SEOUL) >= DEADLINE:
                log("The 16:00 deadline passed; no further work will start.", stream)
                return 1
            return_code = run_command(command, stream, label)
            if return_code:
                failed = True
                break
        if not failed:
            return 0
        if attempt < MAX_PIPELINE_ATTEMPTS:
            log(
                f"Pipeline failed; preserving outputs and retrying in "
                f"{RETRY_DELAY_SECONDS // 60} minutes. See the error above.",
                stream,
            )
            time.sleep(RETRY_DELAY_SECONDS)
    log("All automatic attempts failed; inspect this log for the terminal error.", stream)
    return 1


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as stream:
        log("Scheduled AI Hub training watcher started.", stream)
        log(f"Dataset location: {DATA_ROOT}", stream)
        log(f"First check: {INITIAL_CHECKS[0].isoformat()}", stream)
        log(f"Second check: {INITIAL_CHECKS[1].isoformat()}", stream)
        log(f"Training deadline: {DEADLINE.isoformat()}", stream)

        keep_computer_awake(stream)
        checked: set[datetime] = set()
        while datetime.now(SEOUL) < DEADLINE:
            now = datetime.now(SEOUL)
            due = next((check for check in INITIAL_CHECKS if check not in checked), None)
            if due is not None:
                wait_until(due, stream)
                if datetime.now(SEOUL) >= DEADLINE:
                    break
                checked.add(due)
                log(f"Checking downloaded archives at {due.isoformat()}.", stream)
            else:
                break

            if archives_ready(stream):
                result = run_pipeline(stream)
                log(f"Scheduled run finished with exit code {result}.", stream)
                return result

            if due == INITIAL_CHECKS[0]:
                log("Next scheduled check is at 00:00.", stream)
                continue
            log(
                "The dataset was not complete at either scheduled check; "
                "no files were extracted and no training was started.",
                stream,
            )
            return 1

        log("Deadline reached before a complete dataset was available.", stream)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
