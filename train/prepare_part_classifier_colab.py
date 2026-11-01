import argparse
import io
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CLASS_NAMES = [
    "앞범퍼",
    "앞펜더",
    "전조등",
    "뒷범퍼",
    "뒷펜더",
    "후미등",
    "옆면",
    "사이드미러",
    "스텝",
    "타이어/휠",
]
PART_TO_CLASS = {
    "Front bumper": "앞범퍼",
    "Front fender(L)": "앞펜더",
    "Front fender(R)": "앞펜더",
    "Head lights(L)": "전조등",
    "Head lights(R)": "전조등",
    "Rear bumper": "뒷범퍼",
    "Rear fender(L)": "뒷펜더",
    "Rear fender(R)": "뒷펜더",
    "Rear lamp(L)": "후미등",
    "Rear lamp(R)": "후미등",
    "Front door(L)": "옆면",
    "Front door(R)": "옆면",
    "Rear door(L)": "옆면",
    "Rear door(R)": "옆면",
    "Side mirror(L)": "사이드미러",
    "Side mirror(R)": "사이드미러",
    "Rocker panel(L)": "스텝",
    "Rocker panel(R)": "스텝",
    "Front Wheel(L)": "타이어/휠",
    "Front Wheel(R)": "타이어/휠",
    "Rear Wheel(L)": "타이어/휠",
    "Rear Wheel(R)": "타이어/휠",
}
OVERLAP_THRESHOLD = 0.25


def valid_bbox(value):
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x, y, width, height = (float(item) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def overlap_fraction(damage_box, part_box):
    x, y, width, height = damage_box
    px, py, pwidth, pheight = part_box
    intersection = max(0.0, min(x + width, px + pwidth) - max(x, px)) * max(
        0.0, min(y + height, py + pheight) - max(y, py)
    )
    return intersection / (width * height)


def split_archives(data_root, split):
    source = "1.Training" if split == "train" else "2.Validation"
    image_prefix = "TS" if split == "train" else "VS"
    label_prefix = "TL" if split == "train" else "VL"
    return (
        data_root / source / "1.원천데이터" / f"{image_prefix}_damage_part.zip",
        data_root / source / "2.라벨링데이터" / f"{label_prefix}_damage_part.zip",
    )


def build_records(image_archive_path, label_archive_path, split):
    records_by_image = defaultdict(list)
    report = Counter()
    with zipfile.ZipFile(image_archive_path) as image_zip, zipfile.ZipFile(
        label_archive_path
    ) as label_zip:
        image_members = {
            Path(name).stem.casefold(): name
            for name in image_zip.namelist()
            if Path(name).suffix.casefold()
            in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        }
        label_names = [
            name for name in label_zip.namelist() if name.lower().endswith(".json")
        ]
        report["label_jsons"] = len(label_names)
        for json_index, label_name in enumerate(label_names, start=1):
            if json_index % 10000 == 0:
                print(
                    f"{split} annotations: {json_index:,}/{len(label_names):,}",
                    flush=True,
                )
            image_member = image_members.get(Path(label_name).stem.casefold())
            if image_member is None:
                report["json_without_image"] += 1
                continue
            try:
                annotation_data = json.loads(
                    label_zip.read(label_name).decode("utf-8-sig")
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid annotation JSON {label_name}: {error}") from error
            annotations = annotation_data.get("annotations", [])
            if not isinstance(annotations, list):
                report["invalid_annotations_list"] += 1
                continue

            parts = []
            damages = []
            for annotation_index, annotation in enumerate(annotations):
                if not isinstance(annotation, dict):
                    continue
                box = valid_bbox(annotation.get("bbox"))
                if box is None:
                    continue
                part_name = annotation.get("part")
                if isinstance(part_name, str):
                    mapped_class = PART_TO_CLASS.get(part_name.strip())
                    if mapped_class is not None:
                        parts.append((mapped_class, box))
                damage_name = annotation.get("damage")
                if isinstance(damage_name, str) and damage_name.strip():
                    damages.append((annotation_index, box))

            for annotation_index, damage_box in damages:
                report["damage_objects"] += 1
                candidates = [
                    (overlap_fraction(damage_box, part_box), class_name)
                    for class_name, part_box in parts
                ]
                if not candidates:
                    report["damage_without_part_annotation"] += 1
                    continue
                best_overlap, best_class = max(candidates, key=lambda item: item[0])
                if best_overlap < OVERLAP_THRESHOLD:
                    report["damage_below_overlap_threshold"] += 1
                    continue
                report[f"class::{best_class}"] += 1
                records_by_image[image_member].append(
                    (annotation_index, damage_box, best_class)
                )
    return records_by_image, report


def class_directories():
    return {
        name: f"{index:02d}_{name.replace('/', '-')}"
        for index, name in enumerate(CLASS_NAMES)
    }


def create_crops(records_by_image, image_archive_path, output_root, split):
    directories = class_directories()
    for directory in directories.values():
        (output_root / split / directory).mkdir(parents=True, exist_ok=True)

    counts = Counter(
        {
            class_name: sum(
                1
                for _ in (output_root / split / directory).glob("*.jpg")
            )
            for class_name, directory in directories.items()
        }
    )
    image_items = list(records_by_image.items())
    image_count = len(image_items)
    started = time.monotonic()
    with zipfile.ZipFile(image_archive_path) as image_zip:
        for image_index, (image_member, image_records) in enumerate(
            image_items, start=1
        ):
            pending_records = []
            for annotation_index, damage_box, class_name in image_records:
                output_name = (
                    f"{Path(image_member).stem}__ann{annotation_index:04d}.jpg"
                )
                output_path = (
                    output_root / split / directories[class_name] / output_name
                )
                if not output_path.is_file():
                    pending_records.append(
                        (output_path, damage_box, class_name)
                    )
            if pending_records:
                with Image.open(
                    io.BytesIO(image_zip.read(image_member))
                ) as source_image:
                    source_image = source_image.convert("RGB")
                    image_width, image_height = source_image.size
                    for output_path, (x, y, width, height), class_name in pending_records:
                        center_x, center_y = x + width / 2, y + height / 2
                        crop_width = max(width * 1.5, 48)
                        crop_height = max(height * 1.5, 48)
                        left = max(0, math.floor(center_x - crop_width / 2))
                        top = max(0, math.floor(center_y - crop_height / 2))
                        right = min(image_width, math.ceil(center_x + crop_width / 2))
                        bottom = min(image_height, math.ceil(center_y + crop_height / 2))
                        if right <= left or bottom <= top:
                            counts["invalid_crop"] += 1
                            continue
                        source_image.crop((left, top, right, bottom)).save(
                            output_path, format="JPEG", quality=90
                        )
                        counts[class_name] += 1
            if image_index % 5000 == 0 or image_index == image_count:
                print(
                    f"{split} crops: images {image_index:,}/{image_count:,}; "
                    f"saved {sum(counts[name] for name in CLASS_NAMES):,}; "
                    f"{(time.monotonic() - started) / 60:.1f} min",
                    flush=True,
                )
    return counts


def find_bandizip():
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


def create_archive(output_root, archive_path):
    bandizip = find_bandizip()
    temporary_archive = archive_path.with_name(
        f"{archive_path.stem}.tmp{archive_path.suffix}"
    )
    if temporary_archive.exists():
        temporary_archive.unlink()
    creation = subprocess.run(
        [
            bandizip,
            "c",
            "-l:0",
            "-r",
            "-fmt:zip",
            str(temporary_archive),
            "train",
            "val",
            "dataset_manifest.json",
        ],
        cwd=output_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if creation.returncode != 0:
        raise RuntimeError(
            f"Bandizip create failed ({creation.returncode}):\n"
            f"{creation.stdout}\n{creation.stderr}"
        )
    maximum_size = 10_000_000_000
    if temporary_archive.stat().st_size >= maximum_size:
        temporary_archive.unlink()
        raise ValueError(
            "The complete dataset archive is not below 10 GB; "
            "reduce JPEG quality or sample size before packaging."
        )
    verification = subprocess.run(
        [bandizip, "t", str(temporary_archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    if verification.returncode != 0:
        raise RuntimeError(
            f"Bandizip verification failed ({verification.returncode}):\n"
            f"{verification.stdout}\n{verification.stderr}"
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive.replace(archive_path)
    print(
        f"Bandizip verified: {archive_path} "
        f"({archive_path.stat().st_size / 1e9:.2f} GB)",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build every labeled part-damage crop and package it with Bandizip."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("train/part/160. 차량파손 이미지 데이터/01.데이터"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("train/car_damage_runs/effb3_damage_part_full_crops"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("train/car_damage_runs/car_damage_part_crops_10gb.zip"),
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    archive_path = args.archive.resolve()
    if output_root == data_root or data_root in output_root.parents:
        raise ValueError("Crop output must not be inside or equal to the source dataset.")

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reports = {}
    totals = {}
    for split in ("train", "val"):
        image_archive_path, label_archive_path = split_archives(data_root, split)
        for path in (image_archive_path, label_archive_path):
            if not path.is_file():
                raise FileNotFoundError(f"Source archive not found: {path}")
        records, report = build_records(image_archive_path, label_archive_path, split)
        reports[split] = dict(report)
        saved = create_crops(records, image_archive_path, output_root, split)
        totals[split] = sum(saved[name] for name in CLASS_NAMES)
        expected = sum(report[f"class::{name}"] for name in CLASS_NAMES)
        if totals[split] != expected:
            raise RuntimeError(
                f"{split}: generated {totals[split]:,} crops, expected {expected:,}"
            )
        print(
            f"{split} class counts: "
            + ", ".join(f"{name}={saved[name]:,}" for name in CLASS_NAMES),
            flush=True,
        )

    manifest = {
        "class_names": CLASS_NAMES,
        "train_crops": totals["train"],
        "val_crops": totals["val"],
        "test_mode": False,
        "max_train_jsons": None,
        "max_val_jsons": None,
        "max_train_crops_per_class": None,
        "max_val_crops_per_class": None,
        "overlap_threshold": OVERLAP_THRESHOLD,
        "train_label_report": reports["train"],
        "val_label_report": reports["val"],
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    create_archive(output_root, archive_path)
    print(
        f"Complete: train={totals['train']:,}, val={totals['val']:,}, "
        f"elapsed={(time.monotonic() - started) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
