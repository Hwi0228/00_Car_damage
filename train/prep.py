from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = BASE_DIR / "data" / "extracted"
OUTPUT_DIR = BASE_DIR / "yolo_dataset_sample_95k"
RUN_DIR = BASE_DIR / "car_damage_runs"
RUN_NAME = "aihub_sample_95k"
WEIGHTS = BASE_DIR.parent / "yolov8n-seg.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_MAP = {
    "scratch": 0,
    "scratched": 0,
    "dent": 1,
    "crushed": 1,
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
ANNOTATION_WORKERS = 8


@dataclass(frozen=True)
class ImageLabelPair:
    image_path: Path
    annotation_paths: tuple[Path, ...]
    class_ids: frozenset[int] = frozenset()
    label_lines: tuple[str, ...] = ()
    unsupported_damage: tuple[tuple[str, int], ...] = ()
    invalid_polygons: int = 0


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


def save_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def inspect_pair(pair: ImageLabelPair) -> tuple[ImageLabelPair, Counter[str], int]:
    class_ids: set[int] = set()
    unsupported_damage: Counter[str] = Counter()
    invalid_polygons = 0
    label_lines: list[str] = []
    for annotation_path in pair.annotation_paths:
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
        if not isinstance(filename, str):
            raise ValueError(f"Missing images.file_name in {annotation_path}")
        if Path(filename.replace("\\", "/")).name.casefold() != pair.image_path.name.casefold():
            raise ValueError(
                f"Annotation/image filename mismatch: {annotation_path} -> {filename}"
            )
        width = image_info.get("width")
        height = image_info.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError(f"Invalid image dimensions in {annotation_path}")
        annotations = data.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError(f"Invalid annotations list in {annotation_path}")
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError(f"Invalid annotation entry in {annotation_path}")
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
            class_ids.add(class_id)
            label_lines.extend(f"{class_id} {polygon}" for polygon in polygons)
    return (
        ImageLabelPair(
            pair.image_path,
            pair.annotation_paths,
            frozenset(class_ids),
            tuple(label_lines),
            tuple(unsupported_damage.items()),
            invalid_polygons,
        ),
        unsupported_damage,
        invalid_polygons,
    )


def select_stratified(
    pairs: list[ImageLabelPair], target: int, seed: int
) -> list[ImageLabelPair]:
    grouped: dict[tuple[int, ...], list[ImageLabelPair]] = {}
    for pair in pairs:
        if pair.class_ids:
            key = tuple(sorted(pair.class_ids))
            grouped.setdefault(key, []).append(pair)
    if not grouped:
        raise ValueError("No sampled images contain supported segmentation labels.")

    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)

    class_counts: Counter[int] = Counter()
    selected: list[ImageLabelPair] = []
    class_target = max(1, target // len(CLASS_NAMES))
    while len(selected) < target:
        available = [classes for classes, group in grouped.items() if group]
        if not available:
            break
        scores = {
            classes: sum(
                max(0, class_target - class_counts[class_id]) / class_target
                for class_id in classes
            )
            for classes in available
        }
        best_score = max(scores.values())
        best_groups = [
            classes for classes in available if scores[classes] == best_score
        ]
        chosen_group = rng.choice(best_groups)
        pair = grouped[chosen_group].pop()
        selected.append(pair)
        class_counts.update(pair.class_ids)

    if len(selected) < target:
        print(
            f"Warning: only {len(selected):,} labeled images were available "
            f"for requested sample size {target:,}."
        )
    rng.shuffle(selected)
    return selected


def pair_split(
    split: str,
    target: int,
    seed: int,
    report: dict[str, Any],
    candidate_multiplier: float,
) -> list[ImageLabelPair]:
    image_root = SOURCE_ROOT / split / "images"
    annotation_root = SOURCE_ROOT / split / "annotations"
    images = find_files(image_root, IMAGE_EXTENSIONS)
    annotations = find_files(annotation_root, {".json"})
    image_index: dict[str, Path] = {}
    duplicate_images: list[str] = []
    for image_path in images:
        key = image_path.stem.casefold()
        if key in image_index:
            duplicate_images.append(image_path.name)
        else:
            image_index[key] = image_path

    annotations_by_stem: dict[str, list[Path]] = {}
    duplicate_annotation_stems: list[str] = []
    for annotation_path in annotations:
        key = annotation_path.stem.casefold()
        if key in annotations_by_stem:
            duplicate_annotation_stems.append(annotation_path.name)
        annotations_by_stem.setdefault(key, []).append(annotation_path)

    matched_stems = [
        stem for stem in image_index if stem in annotations_by_stem
    ]
    matched_stem_set = set(matched_stems)
    unmatched_images = set(image_index) - matched_stem_set
    annotations_missing_images = set(annotations_by_stem) - matched_stem_set
    candidates = [
        ImageLabelPair(image_index[key], tuple(annotations_by_stem[key]))
        for key in matched_stems
        if len(annotations_by_stem[key]) == 1
    ]
    if duplicate_images or duplicate_annotation_stems:
        raise ValueError(
            f"{split}: duplicate image/annotation stems found. "
            "Refusing ambiguous pairing."
        )

    pool_size = min(
        len(candidates),
        max(target, math.ceil(target * candidate_multiplier)),
    )
    rng = random.Random(seed)
    candidate_pool = rng.sample(candidates, pool_size)
    checked: list[ImageLabelPair] = []
    unsupported_damage: Counter[str] = Counter()
    invalid_polygons = 0
    malformed_annotations: list[str] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=ANNOTATION_WORKERS) as executor:
        for batch_start in range(0, len(candidate_pool), PROGRESS_INTERVAL):
            batch = candidate_pool[batch_start : batch_start + PROGRESS_INTERVAL]
            results = executor.map(inspect_pair, batch)
            for offset, result in enumerate(results, start=batch_start + 1):
                try:
                    inspected_pair, unsupported, invalid = result
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as error:
                    malformed_annotations.append(
                        f"{candidate_pool[offset - 1].annotation_paths[0].name}: "
                        f"{error}"
                    )
                    continue
                unsupported_damage.update(unsupported)
                invalid_polygons += invalid
                if inspected_pair.class_ids:
                    checked.append(inspected_pair)
                if offset % PROGRESS_INTERVAL == 0 or offset == len(candidate_pool):
                    progress(
                        f"{split} sample scan",
                        offset,
                        len(candidate_pool),
                        started,
                    )
    if candidate_pool:
        sys.stdout.write("\n")

    selected = select_stratified(checked, target, seed)
    selected_class_counts: Counter[str] = Counter()
    for pair in selected:
        selected_class_counts.update(
            CLASS_NAMES[class_id] for class_id in pair.class_ids
        )

    split_report = {
        "images": len(images),
        "annotation_jsons": len(annotations),
        "matched_pairs_by_filename_stem": len(matched_stems),
        "unmatched_images": len(unmatched_images),
        "annotations_missing_images": len(annotations_missing_images),
        "malformed_annotations": len(malformed_annotations),
        "duplicate_image_names": len(duplicate_images),
        "duplicate_annotation_stems": len(duplicate_annotation_stems),
        "candidate_pool_size": len(candidate_pool),
        "valid_labeled_candidates": len(checked),
        "requested_sample_size": target,
        "selected_sample_size": len(selected),
        "selected_class_image_counts": dict(selected_class_counts),
        "unmatched_image_examples": sorted(unmatched_images)[:MAX_REPORTED_NAMES],
        "missing_image_examples": sorted(annotations_missing_images)[
            :MAX_REPORTED_NAMES
        ],
        "malformed_annotation_examples": malformed_annotations[:MAX_REPORTED_NAMES],
        "duplicate_image_examples": duplicate_images[:MAX_REPORTED_NAMES],
    }
    report["splits"][split] = split_report
    print(
        f"{split}: images={len(images):,}, annotations={len(annotations):,}, "
        f"stem_matches={len(matched_stems):,}, selected={len(selected):,}/"
        f"{target:,}, images_without_annotation={len(unmatched_images):,}, "
        f"annotations_without_image={len(annotations_missing_images):,}, "
        f"malformed_sample_annotations={len(malformed_annotations):,}, "
        f"class_image_counts={dict(selected_class_counts)}"
    )
    report.setdefault("unsupported_damage_annotations_in_candidate_pool", {})[
        split
    ] = dict(unsupported_damage)
    report.setdefault("invalid_polygons_in_candidate_pool", {})[split] = (
        invalid_polygons
    )
    return selected


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
            lines = pair.label_lines
            for line in lines:
                class_id = int(line.partition(" ")[0])
                class_counts[CLASS_NAMES[class_id]] += 1
            unsupported_damage.update(dict(pair.unsupported_damage))
            invalid_polygons += pair.invalid_polygons

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


def package_yolo_dataset_for_colab() -> Path:
    archive_path = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_colab.zip"
    bandizip = shutil.which("bz") or shutil.which("bz.exe")
    if bandizip is None:
        raise FileNotFoundError(
            "Bandizip CLI 'bz' was not found on PATH. Install Bandizip or add bz.exe to PATH."
        )
    data_yaml = OUTPUT_DIR / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"YOLO dataset configuration is missing: {data_yaml}")
    for split in ("train", "validation"):
        for kind in ("images", "labels"):
            directory = OUTPUT_DIR / kind / split
            if not directory.is_dir() or not any(directory.iterdir()):
                raise FileNotFoundError(
                    f"YOLO dataset {kind} are missing for {split}: {directory}"
                )

    started = time.monotonic()
    original_yaml = data_yaml.read_bytes()
    data = yaml.safe_load(original_yaml)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YOLO dataset configuration: {data_yaml}")
    data["path"] = f"/content/{OUTPUT_DIR.name}"
    with tempfile.TemporaryDirectory(
        prefix=f".{OUTPUT_DIR.name}_colab_",
        dir=OUTPUT_DIR.parent,
    ) as temporary_directory:
        temporary_archive = Path(temporary_directory) / archive_path.name
        try:
            data_yaml.write_text(
                yaml.safe_dump(data, sort_keys=False),
                encoding="utf-8",
            )
            creation = subprocess.run(
                [
                    bandizip,
                    "c",
                    "-l:0",
                    "-r",
                    "-fmt:zip",
                    str(temporary_archive),
                    "images",
                    "labels",
                    "data.yaml",
                ],
                cwd=OUTPUT_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
            if creation.returncode != 0:
                raise RuntimeError(
                    "Bandizip failed to create the Colab archive "
                    f"(exit {creation.returncode}):\n"
                    f"{creation.stdout}\n{creation.stderr}"
                )

            verification = subprocess.run(
                [bandizip, "t", str(temporary_archive)],
                capture_output=True,
                text=True,
                check=False,
            )
            if verification.returncode != 0:
                raise RuntimeError(
                    "Bandizip reported an invalid Colab archive "
                    f"(exit {verification.returncode}):\n"
                    f"{verification.stdout}\n{verification.stderr}"
                )
            temporary_archive.replace(archive_path)
        finally:
            data_yaml.write_bytes(original_yaml)

    elapsed = time.monotonic() - started
    print(f"Colab upload archive: {archive_path}")
    print(
        f"Archive size: {archive_path.stat().st_size / 10**9:.2f} GB "
        f"({archive_path.stat().st_size / 1024**3:.2f} GiB), "
        f"Bandizip verified in {elapsed:.1f}s"
    )
    return archive_path


def main() -> int:
    global SOURCE_ROOT, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description=(
            "Pair AI Hub images with annotations, create a YOLO segmentation dataset, "
            "and train a deterministic class-balanced sample."
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
        "--zip-for-colab",
        action="store_true",
        help="Create a portable ZIP of the sampled YOLO dataset for Google Drive upload.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Start local training after preparing the sample dataset.",
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
    parser.add_argument(
        "--train-samples",
        type=int,
        default=95000,
        help="Number of class-balanced training images to use.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=23000,
        help="Number of class-balanced validation images to use.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=1.25,
        help="Random candidate-pool size per requested sample, before class balancing.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs for the sampled dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    SOURCE_ROOT = args.source_root.resolve()
    OUTPUT_DIR = args.output_dir.resolve()

    for name, value in (
        ("train-samples", args.train_samples),
        ("val-samples", args.val_samples),
        ("candidate-multiplier", args.candidate_multiplier),
        ("epochs", args.epochs),
    ):
        if value < 1:
            parser.error(f"--{name} must be a positive number.")

    report: dict[str, Any] = {
        "source_root": str(SOURCE_ROOT),
        "sample_seed": args.seed,
        "candidate_multiplier": args.candidate_multiplier,
        "requested_samples": {
            "train": args.train_samples,
            "validation": args.val_samples,
        },
        "splits": {},
    }
    pairs_by_split = {
        "train": pair_split(
            "train",
            args.train_samples,
            args.seed,
            report,
            args.candidate_multiplier,
        ),
        "validation": pair_split(
            "validation",
            args.val_samples,
            args.seed + 1,
            report,
            args.candidate_multiplier,
        ),
    }
    report_path = OUTPUT_DIR / "pairing_report.json"
    save_report(report_path, report)
    print(f"Sample pairing report: {report_path}")
    if args.check_only:
        print("Pair check complete; conversion and training were not started.")
        return 0

    empty_splits = [split for split, pairs in pairs_by_split.items() if not pairs]
    if empty_splits:
        raise ValueError(
            f"No labeled image/annotation pairs in: {', '.join(empty_splits)}. "
            f"See {report_path}."
        )
    data_yaml = create_yolo_dataset(pairs_by_split, report)
    save_report(report_path, report)
    if args.zip_for_colab:
        package_yolo_dataset_for_colab()
    if args.prepare_only:
        print("Sample dataset preparation complete; training was not started.")
        return 0
    if not args.train:
        print("Dataset preparation complete; pass --train to start local training.")
        return 0

    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Pretrained segmentation weights are missing: {WEIGHTS}")

    save_dir = RUN_DIR / RUN_NAME
    last_checkpoint = save_dir / "weights" / "last.pt"
    resuming = last_checkpoint.is_file()
    model = YOLO(str(last_checkpoint if resuming else WEIGHTS))
    print(
        f"Starting {args.epochs}-epoch training on "
        f"{len(pairs_by_split['train']):,} train and "
        f"{len(pairs_by_split['validation']):,} validation images; "
        f"resume={resuming}"
    )
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=320,
        batch=16,
        freeze=10,
        workers=4,
        project=str(RUN_DIR),
        name=RUN_NAME,
        exist_ok=True,
        seed=args.seed,
        amp=True,
        patience=args.epochs,
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
