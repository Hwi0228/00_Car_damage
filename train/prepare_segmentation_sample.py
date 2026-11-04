from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml
from prep import CLASS_MAP, CLASS_NAMES, normalized_polygons


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    ROOT_DIR / "train" / "part" / "160. 차량파손 이미지 데이터" / "01.데이터"
)
OUTPUT_DIR = ROOT_DIR / "train" / "yolo_damage_sample_5k"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    image_member: str
    label_members: tuple[str, ...]
    class_ids: tuple[int, ...]


def archive_paths(split: str) -> tuple[Path, Path]:
    if split == "train":
        source_dir, image_prefix, label_prefix = "1.Training", "TS", "TL"
    else:
        source_dir, image_prefix, label_prefix = "2.Validation", "VS", "VL"
    split_dir = SOURCE_DIR / source_dir
    return (
        split_dir / "1.원천데이터" / f"{image_prefix}_damage_part.zip",
        split_dir / "2.라벨링데이터" / f"{label_prefix}_damage_part.zip",
    )


def inspect_split(split: str) -> tuple[list[ImageRecord], Counter[int]]:
    image_archive_path, label_archive_path = archive_paths(split)
    for path in (image_archive_path, label_archive_path):
        if not path.is_file():
            raise FileNotFoundError(f"AIHub source archive is missing: {path}")

    records: list[ImageRecord] = []
    image_class_counts: Counter[int] = Counter()
    with zipfile.ZipFile(image_archive_path) as image_archive, zipfile.ZipFile(
        label_archive_path
    ) as label_archive:
        images_by_stem: dict[str, str] = {}
        for name in image_archive.namelist():
            member = PurePosixPath(name)
            if member.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            stem = member.stem.casefold()
            if stem in images_by_stem:
                raise ValueError(f"{split}: duplicate image filename stem {stem}")
            images_by_stem[stem] = name

        labels_by_stem: dict[str, list[str]] = defaultdict(list)
        for name in label_archive.namelist():
            if not name.casefold().endswith(".json"):
                continue
            stem = PurePosixPath(name).stem.casefold()
            labels_by_stem[stem].append(name)

        missing_images = labels_by_stem.keys() - images_by_stem.keys()
        if missing_images:
            raise FileNotFoundError(
                f"{split}: {len(missing_images):,} annotations have no matching image; "
                f"examples: {sorted(missing_images)[:5]}"
            )

        stems = sorted(labels_by_stem)
        for index, stem in enumerate(stems, start=1):
            image_member = images_by_stem[stem]
            label_members = tuple(sorted(labels_by_stem[stem]))
            class_ids: set[int] = set()
            for label_member in label_members:
                try:
                    annotation_data = json.loads(
                        label_archive.read(label_member).decode("utf-8-sig")
                    )
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid annotation JSON {label_member}: {error}"
                    ) from error
                image_info = annotation_data.get("images", {})
                if isinstance(image_info, list):
                    image_info = image_info[0] if image_info else {}
                if not isinstance(image_info, dict):
                    raise ValueError(f"Invalid image metadata in {label_member}")
                width, height = image_info.get("width"), image_info.get("height")
                if not isinstance(width, int) or not isinstance(height, int):
                    raise ValueError(f"Invalid image dimensions in {label_member}")
                for annotation in annotation_data.get("annotations", []):
                    if not isinstance(annotation, dict):
                        continue
                    damage = annotation.get("damage")
                    class_id = (
                        CLASS_MAP.get(damage.strip().casefold())
                        if isinstance(damage, str)
                        else None
                    )
                    if class_id is None:
                        continue
                    if normalized_polygons(
                        annotation.get("segmentation"), width, height
                    ):
                        class_ids.add(class_id)

            if class_ids:
                signature = tuple(sorted(class_ids))
                records.append(ImageRecord(image_member, label_members, signature))
                image_class_counts.update(class_ids)

            if index % 20_000 == 0 or index == len(stems):
                print(
                    f"{split}: scanned {index:,}/{len(stems):,} image labels",
                    flush=True,
                )

    if not records:
        raise ValueError(f"{split}: no images have supported damage polygons.")
    return records, image_class_counts


def allocate_strata(
    groups: dict[tuple[int, ...], list[ImageRecord]], target: int
) -> dict[tuple[int, ...], int]:
    total = sum(len(records) for records in groups.values())
    if target > total:
        raise ValueError(f"Requested {target:,} images but only {total:,} are labeled.")

    exact = {key: target * len(records) / total for key, records in groups.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remainder = target - sum(quotas.values())
    order = sorted(
        groups,
        key=lambda key: (exact[key] - quotas[key], len(groups[key]), key),
        reverse=True,
    )
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def select_representative(
    records: list[ImageRecord], target: int, seed: int
) -> list[ImageRecord]:
    groups: dict[tuple[int, ...], list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.class_ids].append(record)

    quotas = allocate_strata(groups, target)
    rng = random.Random(seed)
    selected: list[ImageRecord] = []
    for signature in sorted(groups):
        selected.extend(rng.sample(groups[signature], quotas[signature]))
    rng.shuffle(selected)
    return selected


def read_label_lines(label_archive: zipfile.ZipFile, record: ImageRecord) -> list[str]:
    lines: list[str] = []
    for label_member in record.label_members:
        annotation_data = json.loads(
            label_archive.read(label_member).decode("utf-8-sig")
        )
        image_info = annotation_data.get("images", {})
        if isinstance(image_info, list):
            image_info = image_info[0] if image_info else {}
        width, height = image_info["width"], image_info["height"]
        for annotation in annotation_data.get("annotations", []):
            if not isinstance(annotation, dict):
                continue
            damage = annotation.get("damage")
            class_id = (
                CLASS_MAP.get(damage.strip().casefold())
                if isinstance(damage, str)
                else None
            )
            if class_id is None:
                continue
            polygons = normalized_polygons(
                annotation.get("segmentation"), width, height
            )
            lines.extend(f"{class_id} {polygon}" for polygon in polygons)
    return lines


def materialize_split(
    split: str,
    selected: list[ImageRecord],
    output_dir: Path,
) -> tuple[Counter[int], Counter[int]]:
    image_archive_path, label_archive_path = archive_paths(split)
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_counts: Counter[int] = Counter()
    instance_counts: Counter[int] = Counter()
    seen_names: set[str] = set()

    with zipfile.ZipFile(image_archive_path) as image_archive, zipfile.ZipFile(
        label_archive_path
    ) as label_archive:
        for index, record in enumerate(selected, start=1):
            filename = PurePosixPath(record.image_member).name
            if filename.casefold() in seen_names:
                raise ValueError(f"{split}: duplicate selected image filename {filename}")
            seen_names.add(filename.casefold())

            image_path = image_dir / filename
            label_path = label_dir / f"{Path(filename).stem}.txt"
            image_path.write_bytes(image_archive.read(record.image_member))
            label_lines = read_label_lines(label_archive, record)
            if not label_lines:
                raise ValueError(
                    f"Selected image no longer has valid labels: {record.label_members}"
                )
            label_path.write_text("\n".join(label_lines), encoding="utf-8")
            image_counts.update(record.class_ids)
            instance_counts.update(
                int(line.partition(" ")[0]) for line in label_lines
            )
            if index % 1_000 == 0 or index == len(selected):
                print(f"{split}: wrote {index:,}/{len(selected):,} images", flush=True)

    return image_counts, instance_counts


def find_bandizip() -> str:
    candidates = [
        shutil.which("bz"),
        shutil.which("bz.exe"),
        str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Bandizip" / "bz.exe"),
        str(
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
            / "Bandizip"
            / "bz.exe"
        ),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("Bandizip CLI 'bz.exe' was not found.")


def create_colab_archive(output_dir: Path) -> Path:
    archive_path = output_dir.with_name(f"{output_dir.name}_colab.zip")
    temporary_archive = archive_path.with_name(
        f"{archive_path.stem}.tmp{archive_path.suffix}"
    )
    if temporary_archive.exists():
        temporary_archive.unlink()
    bandizip = find_bandizip()
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
            "pairing_report.json",
        ],
        cwd=output_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if creation.returncode != 0:
        raise RuntimeError(
            f"Bandizip failed to create {temporary_archive} "
            f"(exit {creation.returncode}):\n{creation.stdout}\n{creation.stderr}"
        )
    verification = subprocess.run(
        [bandizip, "t", str(temporary_archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    if verification.returncode != 0:
        raise RuntimeError(
            f"Bandizip failed to verify {temporary_archive} "
            f"(exit {verification.returncode}):\n"
            f"{verification.stdout}\n{verification.stderr}"
        )
    temporary_archive.replace(archive_path)
    print(
        f"Verified Colab ZIP: {archive_path} "
        f"({archive_path.stat().st_size / 1e9:.2f} GB)",
        flush=True,
    )
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a representative YOLO damage-segmentation sample directly "
            "from the AIHub ZIP archives under train/part."
        )
    )
    parser.add_argument("--train-samples", type=int, default=4_000)
    parser.add_argument("--val-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--zip-for-colab",
        action="store_true",
        help="Create and verify a Bandizip ZIP for Colab upload.",
    )
    args = parser.parse_args()
    if args.train_samples < 1 or args.val_samples < 1:
        parser.error("--train-samples and --val-samples must be positive.")
    output_dir = args.output_dir.resolve()
    if output_dir == SOURCE_DIR.resolve() or SOURCE_DIR.resolve() in output_dir.parents:
        parser.error("Output directory must not be inside the source archive directory.")

    started = time.monotonic()
    all_records: dict[str, list[ImageRecord]] = {}
    all_source_counts: dict[str, Counter[int]] = {}
    selected: dict[str, list[ImageRecord]] = {}
    for split, target, seed in (
        ("train", args.train_samples, args.seed),
        ("validation", args.val_samples, args.seed + 1),
    ):
        records, source_counts = inspect_split(split)
        all_records[split] = records
        all_source_counts[split] = source_counts
        selected[split] = select_representative(records, target, seed)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir to preserve existing files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_stats = {}
    for split in ("train", "validation"):
        image_counts, instance_counts = materialize_split(
            split, selected[split], output_dir
        )
        selected_stats[split] = {
            "selected_images": len(selected[split]),
            "source_labeled_images": len(all_records[split]),
            "source_class_image_counts": {
                CLASS_NAMES[class_id]: all_source_counts[split][class_id]
                for class_id in range(len(CLASS_NAMES))
            },
            "selected_class_image_counts": {
                CLASS_NAMES[class_id]: image_counts[class_id]
                for class_id in range(len(CLASS_NAMES))
            },
            "selected_class_polygon_counts": {
                CLASS_NAMES[class_id]: instance_counts[class_id]
                for class_id in range(len(CLASS_NAMES))
            },
        }

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output_dir),
                "train": "images/train",
                "val": "images/validation",
                "names": {
                    class_id: class_name
                    for class_id, class_name in CLASS_NAMES.items()
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = {
        "source": "AIHub damage_part archives under train/part",
        "selection": "proportional stratified sampling by per-image damage-class combination",
        "seed": args.seed,
        "requested_images": {
            "train": args.train_samples,
            "validation": args.val_samples,
            "total": args.train_samples + args.val_samples,
        },
        "splits": selected_stats,
    }
    (output_dir / "pairing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Selected class-image and polygon counts:")
    for split, stats in selected_stats.items():
        print(
            split,
            "images=",
            stats["selected_images"],
            "class_images=",
            stats["selected_class_image_counts"],
            "polygons=",
            stats["selected_class_polygon_counts"],
        )

    if args.zip_for_colab:
        create_colab_archive(output_dir)
    print(
        f"Complete: {output_dir} "
        f"({time.monotonic() - started:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
