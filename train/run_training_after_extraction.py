from __future__ import annotations

import base64
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
EXTRACTED_ROOT = DATA_ROOT / "extracted"
LOG_PATH = BASE_DIR / "scheduled_training.log"
SEOUL = ZoneInfo("Asia/Seoul")
DEADLINE = datetime(2026, 10, 29, 16, 0, tzinfo=SEOUL)
EXPECTED_OUTPUTS = (
    EXTRACTED_ROOT / "train" / "images",
    EXTRACTED_ROOT / "train" / "annotations",
    EXTRACTED_ROOT / "validation" / "images",
    EXTRACTED_ROOT / "validation" / "annotations",
)
POLL_SECONDS = 60
RETRY_SECONDS = 300
MAX_ATTEMPTS = 3


def log(message: str, stream: TextIO) -> None:
    line = f"[{datetime.now(SEOUL).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


def keep_computer_awake(stream: TextIO) -> None:
    if sys.platform != "win32":
        return
    try:
        flags = 0x80000000 | 0x00000001
        if not ctypes.windll.kernel32.SetThreadExecutionState(flags):
            log("WARNING: Windows could not be kept awake.", stream)
    except (AttributeError, OSError) as error:
        log(f"WARNING: Could not prevent system sleep: {error}", stream)


def archives_available() -> bool:
    for _, _, name in extract_dataset.ARCHIVES:
        try:
            path = extract_dataset.find_archive(name)
            if not path.is_file() or path.stat().st_size == 0:
                return False
            extract_dataset.archive_stats(path)
        except (OSError, ValueError, zipfile.BadZipFile):
            return False
    return True


def extractor_running() -> bool:
    if sys.platform != "win32":
        return False
    command = (
        "$p=Get-CimInstance Win32_Process | Where-Object { "
        "($_.Name -eq 'python.exe' -and $_.CommandLine -match 'extract_dataset\\.py') "
        "-or ($_.Name -eq 'tar.exe' -and $_.CommandLine -match "
        "'(TS|TL|VS|VL)_damage\\.zip') }; "
        "if ($p) { 'RUNNING' } else { 'IDLE' }"
    )
    encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return "RUNNING" in result.stdout


def outputs_complete() -> bool:
    return all(
        directory.is_dir() and any(directory.iterdir())
        for directory in EXPECTED_OUTPUTS
    )


def run_command(
    command: list[str], stream: TextIO, label: str, cwd: Path
) -> int:
    log(f"Starting {label}: {' '.join(command)}", stream)
    process = subprocess.Popen(
        command,
        cwd=cwd,
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


def wait_for_extraction(stream: TextIO) -> bool:
    while datetime.now(SEOUL) < DEADLINE:
        if outputs_complete():
            log("All four extracted image/annotation folders are present.", stream)
            return True
        if extractor_running():
            log(
                "Dataset extraction is still running; waiting without starting training.",
                stream,
            )
            time.sleep(POLL_SECONDS)
            continue
        if not archives_available():
            log(
                "Extraction is not running and complete source ZIPs are unavailable; "
                "waiting without modifying data.",
                stream,
            )
            time.sleep(POLL_SECONDS)
            continue

        log(
            "Source ZIPs are available but extraction is incomplete; "
            "resuming the verified sequential extractor.",
            stream,
        )
        result = run_command(
            [sys.executable, "-u", str(BASE_DIR / "extract_dataset.py")],
            stream,
            "dataset extraction",
            BASE_DIR.parent,
        )
        if result:
            log(
                "Extraction failed. Existing files are preserved; "
                f"retrying in {RETRY_SECONDS // 60} minutes.",
                stream,
            )
            time.sleep(RETRY_SECONDS)
    log("Deadline reached before extraction completed.", stream)
    return False


def main() -> int:
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as stream:
        log("Scheduled train-after-extraction task started.", stream)
        log(f"Dataset root: {DATA_ROOT}", stream)
        log(f"Training deadline: {DEADLINE.isoformat()}", stream)
        keep_computer_awake(stream)

        if not wait_for_extraction(stream):
            return 1

        for attempt in range(1, MAX_ATTEMPTS + 1):
            if datetime.now(SEOUL) >= DEADLINE:
                log("The training deadline has passed; not starting another attempt.", stream)
                return 1
            log(f"Pair check / preparation / training attempt {attempt}/{MAX_ATTEMPTS}.", stream)
            result = run_command(
                [sys.executable, "-u", str(BASE_DIR / "prep.py")],
                stream,
                "pair validation and deadline-bounded training",
                BASE_DIR.parent,
            )
            if result == 0:
                log("Training pipeline completed.", stream)
                return 0
            if attempt < MAX_ATTEMPTS:
                log(
                    "Pipeline exited with an error; see the log above. "
                    "Preserving generated data and checkpoint, then retrying in "
                    f"{RETRY_SECONDS // 60} minutes.",
                    stream,
                )
                time.sleep(RETRY_SECONDS)
        log("Automatic retries exhausted; inspect the logged terminal error.", stream)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
