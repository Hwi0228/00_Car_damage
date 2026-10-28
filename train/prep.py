from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = BASE_DIR / "data" / "extracted"
OUTPUT_DIR = BASE_DIR / "yolo_dataset_full"
RUN_DIR = BASE_DIR / "car_damage_runs"
RUN_NAME = "aihub_full_until_20261029"
WEIGHTS = BASE_DIR.parent / "yolov8n-seg.pt"
SEOUL = ZoneInfo("Asia/Seoul")
DEADLINE = datetime(2026, 10, 29, 16, 0, tzinfo=SEOUL)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_MAP = {
    "scratch": 0,
    "dent": 1,
    "breakage": 2,
    "separated": 3,
    "separation": 3,
    "스크래치": 0,
    "찌그러짐": 1,
    "파손": 2,
    "이격": 3,
}
CLASS_NAMES = {0: "Scratch", 1: "Dent", 2: "Breakage", 3: "Separated"}
PROGRESS_INTERVAL = 1000
MAX_REPORTED_NAMES = 100


@dataclass(frozen=True)
class ImageLabelPair:
    image_path: Path
    annotation_paths: tuple[Path, ...]


def progress(label: str, current: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 0.001)
    rate = current / elapsed
    percent = current * 100 / total if total else 100.0
    width = 24
    filled = round(width * min(current / total, 1)) if total else width
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\r{label}: [{bar}] {percent:5.1f}% "
        f"{current:,}/{total:,} - {rate:,.0f} files/s"
    )
    sys.stdout.flush()


def find_files(root: Path, extensions: set[str]) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder is missing: {root}")
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in extensions
    ]
    return sorted(files)


def metadata_filename(annotation_path: Path) -> str:
    with annotation_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    image_info = data.get("images", {})
    if isinstance(image_info, list):
        if not image_info:
            raise ValueError(f"No images metadata in {annotation_path}")
        image_info = image_info[0]
    if not isinstance(image_info, dict):
        raise ValueError(f"Invalid images metadata in {annotation_path}")
    filename = image_info.get("file_name")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"Missing images.file_name in {annotation_path}")
    filename = filename.strip().replace("\\", "/")
    member = PurePosixPath(filename)
    if member.is_absolute() or len(member.parts) != 1 or member.name in {".", ".."}:
        raise ValueError(f"Unsafe images.file_name in {annotation_path}: {filename}")
    return member.name


def save_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def pair_split(split: str, report: dict[str, Any]) -> list[ImageLabelPair]:
    image_root = SOURCE_ROOT / split / "images"
    annotation_root = SOURCE_ROOT / split / "annotations"
    images = find_files(image_root, IMAGE_EXTENSIONS)
    annotations = find_files(annotation_root, {".json"})
    image_index: dict[str, Path] = {}
    duplicate_images: list[str] = []
    for image_path in images:
        key = image_path.name.casefold()
        if key in image_index:
            duplicate_images.append(image_path.name)
        else:
            image_index[key] = image_path

    annotations_by_image: dict[str, list[Path]] = {}
    missing_images: list[str] = []
    malformed_annotations: list[str] = []
    filename_disagreements: list[str] = []
    duplicate_label_stems: list[str] = []
    started = time.monotonic()
    for index, annotation_path in enumerate(annotations, start=1):
        try:
            image_name = metadata_filename(annotation_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            malformed_annotations.append(
                f"{annotation_path.relative_to(annotation_root)}: {error}"
            )
            continue

        key = image_name.casefold()
        image_path = image_index.get(key)
        if image_path is None:
            missing_images.append(image_name)
        else:
            if annotation_path.stem.casefold() != image_path.stem.casefold():
                filename_disagreements.append(
                    f"{annotation_path.name} -> {image_path.name}"
                )
            annotations_by_image.setdefault(key, []).append(annotation_path)

        if index % PROGRESS_INTERVAL == 0 or index == len(annotations):
            progress(f"{split} pairing", index, len(annotations), started)
    if annotations:
        sys.stdout.write("\n")

    unmatched_images = sorted(
        path.name
        for path in images
        if path.name.casefold() not in annotations_by_image
    )
    duplicate_references = [
        image_index[key].name
        for key, paths in annotations_by_image.items()
        if len(paths) > 1
    ]
    image_stems: set[str] = set()
    pairs: list[ImageLabelPair] = []
    for image_path in images:
        key = image_path.name.casefold()
        annotation_paths = annotations_by_image.get(key)
        if not annotation_paths:
            continue
        stem = image_path.stem.casefold()
        if stem in image_stems:
            duplicate_label_stems.append(image_path.stem)
        image_stems.add(stem)
        pairs.append(ImageLabelPair(image_path, tuple(annotation_paths)))

    split_report = {
        "images": len(images),
        "annotation_jsons": len(annotations),
        "matched_pairs": len(pairs),
        "unmatched_images": len(unmatched_images),
        "annotations_missing_images": len(missing_images),
        "malformed_annotations": len(malformed_annotations),
        "duplicate_image_names": len(duplicate_images),
        "images_with_multiple_annotations": len(duplicate_references),
        "extra_annotations_for_same_image": sum(
            max(0, len(paths) - 1) for paths in annotations_by_image.values()
        ),
        "duplicate_label_stems": len(duplicate_label_stems),
        "json_filename_disagreements": len(filename_disagreements),
        "unmatched_image_examples": unmatched_images[:MAX_REPORTED_NAMES],
        "missing_image_examples": missing_images[:MAX_REPORTED_NAMES],
        "malformed_annotation_examples": malformed_annotations[:MAX_REPORTED_NAMES],
        "duplicate_image_examples": duplicate_images[:MAX_REPORTED_NAMES],
        "multiple_annotation_image_examples": duplicate_references[
            :MAX_REPORTED_NAMES
        ],
        "duplicate_label_stem_examples": duplicate_label_stems[:MAX_REPORTED_NAMES],
        "json_filename_disagreement_examples": filename_disagreements[
            :MAX_REPORTED_NAMES
        ],
    }
    report["splits"][split] = split_report
    print(
        f"{split}: images={len(images):,}, annotations={len(annotations):,}, "
        f"matched={len(pairs):,}, images_without_annotation={len(unmatched_images):,}, "
        f"annotations_without_image={len(missing_images):,}, "
        f"malformed={len(malformed_annotations):,}"
    )
    return pairs


def polygon_candidates(segmentation: Any) -> list[list[tuple[float, float]]]:
    if not isinstance(segmentation, list) or not segmentation:
        return []
    if all(isinstance(value, (int, float)) for value in segmentation):
        if len(segmentation) < 6 or len(segmentation) % 2:
            return []
        return [
            [
                (float(segmentation[index]), float(segmentation[index + 1]))
                for index in range(0, len(segmentation), 2)
            ]
        ]
    if all(
        isinstance(point, list)
        and len(point) >= 2
        and isinstance(point[0], (int, float))
        and isinstance(point[1], (int, float))
        for point in segmentation
    ):
        return [[(float(point[0]), float(point[1])) for point in segmentation]]
    polygons: list[list[tuple[float, float]]] = []
    for child in segmentation:
        polygons.extend(polygon_candidates(child))
    return polygons


def normalized_polygons(
    segmentation: Any, image_width: int, image_height: int
) -> list[str]:
    if image_width <= 0 or image_height <= 0:
        return []
    result = []
    for polygon in polygon_candidates(segmentation):
        if len(polygon) < 3 or any(
            not math.isfinite(x) or not math.isfinite(y) for x, y in polygon
        ):
            continue
        coordinates = []
        for x, y in polygon:
            coordinates.extend(
                (
                    f"{max(0.0, min(1.0, x / image_width)):.6f}",
                    f"{max(0.0, min(1.0, y / image_height)):.6f}",
                )
            )
        result.append(" ".join(coordinates))
    return result


def create_yolo_dataset(
    pairs_by_split: dict[str, list[ImageLabelPair]], report: dict[str, Any]
) -> Path:
    class_counts: Counter[str] = Counter()
    unsupported_damage: Counter[str] = Counter()
    invalid_polygons = 0
    empty_labels: Counter[str] = Counter()
    free_bytes = shutil.disk_usage(OUTPUT_DIR.parent).free
    required_bytes = 2 * 1024**3
    print(
        "YOLO dataset will hard-link source images to avoid duplicating the "
        f"large image files; free={free_bytes / 1024**3:.2f} GiB"
    )
    if free_bytes < required_bytes:
        raise OSError(
            f"Not enough disk space for dataset copies. Need at least "
            f"{required_bytes / 1024**3:.2f} GiB free."
        )

    for split, pairs in pairs_by_split.items():
        image_dir = OUTPUT_DIR / "images" / split
        label_dir = OUTPUT_DIR / "labels" / split
        expected_images = {pair.image_path.name for pair in pairs}
        expected_labels = {f"{pair.image_path.stem}.txt" for pair in pairs}
        for directory in (image_dir, label_dir):
            expected = expected_images if directory == image_dir else expected_labels
            existing = (
                {path.name for path in directory.iterdir() if path.is_file()}
                if directory.exists()
                else set()
            )
            unexpected = existing - expected
            if unexpected:
                raise FileExistsError(
                    f"{directory} contains files not in the matched source data: "
                    f"{sorted(unexpected)[:5]}. Inspect them before retrying."
                )
            directory.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        for index, pair in enumerate(pairs, start=1):
            lines: list[str] = []
            for annotation_path in pair.annotation_paths:
                with annotation_path.open("r", encoding="utf-8-sig") as file:
                    data = json.load(file)
                image_info = data.get("images", {})
                if isinstance(image_info, list):
                    image_info = image_info[0]
                width = image_info.get("width")
                height = image_info.get("height")
                if not isinstance(width, int) or not isinstance(height, int):
                    raise ValueError(f"Invalid image dimensions in {annotation_path}")

                annotations = data.get("annotations", [])
                if not isinstance(annotations, list):
                    raise ValueError(f"Invalid annotations list in {annotation_path}")
                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        raise ValueError(
                            f"Invalid annotation entry in {annotation_path}"
                        )
                    damage = annotation.get("damage")
                    if not isinstance(damage, str):
                        unsupported_damage[str(damage)] += 1
                        continue
                    class_id = CLASS_MAP.get(damage.strip().casefold())
                    if class_id is None:
                        unsupported_damage[damage] += 1
                        continue
                    polygons = normalized_polygons(
                        annotation.get("segmentation"), width, height
                    )
                    if not polygons:
                        invalid_polygons += 1
                        continue
                    class_counts[CLASS_NAMES[class_id]] += 1
                    lines.extend(f"{class_id} {polygon}" for polygon in polygons)

            if not lines:
                empty_labels[split] += 1
            image_destination = image_dir / pair.image_path.name
            label_destination = label_dir / f"{pair.image_path.stem}.txt"
            try:
                image_destination.hardlink_to(pair.image_path)
            except FileExistsError:
                if not os.path.samefile(image_destination, pair.image_path):
                    raise OSError(
                        f"Existing YOLO image is not linked to its source: "
                        f"{image_destination}"
                    )
            except OSError as error:
                raise OSError(
                    f"Could not hard-link {pair.image_path} to {image_destination}. "
                    "Both directories must be on the same local NTFS volume."
                ) from error
            label_destination.write_text("\n".join(lines), encoding="utf-8")
            if index % PROGRESS_INTERVAL == 0 or index == len(pairs):
                progress(f"{split} conversion", index, len(pairs), started)
        sys.stdout.write("\n")

    if not class_counts:
        raise ValueError("No valid segmentation polygons were found in the matched data.")
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(OUTPUT_DIR.resolve()),
                "train": "images/train",
                "val": "images/validation",
                "names": CLASS_NAMES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report["class_instances"] = dict(class_counts)
    report["unsupported_damage_annotations"] = dict(unsupported_damage)
    report["invalid_segmentation_annotations"] = invalid_polygons
    report["images_with_no_valid_polygons"] = dict(empty_labels)
    report["yolo_dataset"] = str(OUTPUT_DIR)
    return yaml_path


def main() -> int:
    global SOURCE_ROOT, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description=(
            "Pair AI Hub images with annotations, create a YOLO segmentation dataset, "
            "and train until the fixed Seoul deadline."
        )
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check image/annotation pairing and write a report without converting or training.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Check pairings and create the YOLO dataset without starting training.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SOURCE_ROOT,
        help="Root containing train/images, train/annotations, validation/images, and validation/annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Generated YOLO dataset directory.",
    )
    args = parser.parse_args()
    SOURCE_ROOT = args.source_root.resolve()
    OUTPUT_DIR = args.output_dir.resolve()

    now = datetime.now(SEOUL)
    if now >= DEADLINE:
        print(f"Training deadline has passed: {DEADLINE.isoformat()}")
        return 0

    if not args.check_only and (OUTPUT_DIR / "data.yaml").is_file():
        existing_pairs: dict[str, set[str]] = {}
        for split in ("train", "validation"):
            image_dir = OUTPUT_DIR / "images" / split
            label_dir = OUTPUT_DIR / "labels" / split
            if not image_dir.is_dir() or not label_dir.is_dir():
                break
            image_stems = {
                path.stem.casefold()
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            }
            label_stems = {
                path.stem.casefold()
                for path in label_dir.iterdir()
                if path.is_file() and path.suffix.casefold() == ".txt"
            }
            if not image_stems or image_stems != label_stems:
                break
            existing_pairs[split] = image_stems
        existing_config = yaml.safe_load(
            (OUTPUT_DIR / "data.yaml").read_text(encoding="utf-8")
        )
        expected_config = {
            "path": str(OUTPUT_DIR),
            "train": "images/train",
            "val": "images/validation",
            "names": CLASS_NAMES,
        }
        if len(existing_pairs) == 2 and existing_config == expected_config:
            data_yaml = OUTPUT_DIR / "data.yaml"
            print(
                "Using the existing verified YOLO dataset; skipping source "
                "pairing and conversion so training can resume without recopying."
            )
        else:
            data_yaml = None
    else:
        data_yaml = None

    report: dict[str, Any] = {
        "source_root": str(SOURCE_ROOT),
        "checked_at": now.isoformat(),
        "deadline": DEADLINE.isoformat(),
        "splits": {},
    }
    if data_yaml is None:
        pairs_by_split = {
            split: pair_split(split, report) for split in ("train", "validation")
        }
        report_path = OUTPUT_DIR / "pairing_report.json"
        save_report(report_path, report)
        print(f"Pairing report: {report_path}")
        if args.check_only:
            print("Pair check complete; conversion and training were not started.")
            return 0
        duplicate_counts = {
            split: {
                key: details[key]
                for key in (
                    "duplicate_image_names",
                    "duplicate_label_stems",
                )
                if details[key]
            }
            for split, details in report["splits"].items()
        }
        duplicate_counts = {
            split: counts for split, counts in duplicate_counts.items() if counts
        }
        if duplicate_counts:
            raise ValueError(
                f"Duplicate image names or label stems found: {duplicate_counts}. "
                f"See {report_path}."
            )
        empty_splits = [
            split
            for split, pairs in pairs_by_split.items()
            if not pairs
        ]
        if empty_splits:
            raise ValueError(
                f"No matching image/annotation pairs in: {', '.join(empty_splits)}. "
                f"See {report_path}."
            )
        data_yaml = create_yolo_dataset(pairs_by_split, report)
        save_report(report_path, report)
        if args.prepare_only:
            print("Dataset preparation complete; training was not started.")
            return 0
    elif args.check_only:
        raise FileNotFoundError(
            "The original source folders are needed for --check-only; "
            "the existing YOLO dataset alone cannot verify source image/JSON pairing."
        )

    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Pretrained segmentation weights are missing: {WEIGHTS}")

    remaining_seconds = (DEADLINE - datetime.now(SEOUL)).total_seconds()
    if remaining_seconds <= 0:
        print("The deadline passed during dataset preparation; training was not started.")
        return 0

    save_dir = RUN_DIR / RUN_NAME
    last_checkpoint = save_dir / "weights" / "last.pt"
    resuming = last_checkpoint.is_file()
    model = YOLO(str(last_checkpoint if resuming else WEIGHTS))

    def stop_at_deadline(trainer: Any) -> None:
        if datetime.now(SEOUL) >= DEADLINE:
            trainer.stop = True
            print(
                f"\nDeadline reached ({DEADLINE.isoformat()}); "
                "stopping after this batch and saving the checkpoint."
            )

    model.add_callback("on_train_batch_end", stop_at_deadline)
    print(
        f"Starting training at {datetime.now(SEOUL).isoformat()} "
        f"until {DEADLINE.isoformat()}; resume={resuming}"
    )
    model.train(
        data=str(data_yaml),
        epochs=10000,
        time=remaining_seconds / 3600,
        imgsz=320,
        batch=16,
        freeze=10,
        workers=2,
        project=str(RUN_DIR),
        name=RUN_NAME,
        exist_ok=True,
        seed=0,
        amp=True,
        patience=10000,
        save=True,
        save_period=-1,
        resume=str(last_checkpoint) if resuming else False,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
