from __future__ import annotations

import argparse
import stat
import zipfile
from pathlib import Path, PurePosixPath

import yaml
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT_DIR / "train"
DEFAULT_DATA = TRAIN_DIR / "loaded kor coin.v2i.yolov8.zip"
DEFAULT_MODEL = ROOT_DIR / "yolov8n.pt"
DATASET_DIR = TRAIN_DIR / "coin_dataset"
DEFAULT_PROJECT = TRAIN_DIR / "coin_runs"


def extract_dataset(archive_path: Path) -> Path:
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Dataset archive is not a readable ZIP: {archive_path}")
    if DATASET_DIR.is_dir() and (DATASET_DIR / "data.yaml").is_file():
        return DATASET_DIR
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        raise FileExistsError(
            f"Dataset destination is not empty: {DATASET_DIR}. "
            "Move or rename it before extracting the archive again."
        )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    root = DATASET_DIR.resolve()
    seen: dict[str, tuple[int, int]] = {}

    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename.replace("\\", "/"))
            if (
                member.is_absolute()
                or not member.parts
                or any(part in {"", ".", ".."} for part in member.parts)
                or ":" in member.parts[0]
            ):
                raise ValueError(f"Unsafe path in dataset archive: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Symbolic links are not allowed: {info.filename}")

            target = (DATASET_DIR / Path(*member.parts)).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in dataset archive: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            signature = (info.CRC, info.file_size)
            if info.filename in seen:
                if seen[info.filename] != signature:
                    raise ValueError(
                        f"Conflicting duplicate files in dataset archive: {info.filename}"
                    )
                continue
            seen[info.filename] = signature
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                output.write(source.read())

    return DATASET_DIR


def prepare_data_yaml(data_path: Path) -> Path:
    if data_path.suffix.lower() not in {".zip", ".yaml", ".yml"}:
        raise ValueError("Dataset must be a ZIP archive or a YOLO data.yaml.")
    if data_path.suffix.lower() in {".yaml", ".yml"}:
        return data_path

    dataset_root = extract_dataset(data_path)
    source_yaml = dataset_root / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Dataset archive has no root data.yaml: {data_path}")
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or not isinstance(config.get("names"), (list, dict)):
        raise ValueError(f"Invalid YOLO dataset configuration: {source_yaml}")

    split_directories = {
        "train": ("train",),
        "val": ("valid", "val"),
        "test": ("test",),
    }
    for split, candidates in split_directories.items():
        image_dir = next(
            (
                dataset_root / candidate / "images"
                for candidate in candidates
                if (dataset_root / candidate / "images").is_dir()
            ),
            None,
        )
        if image_dir is None:
            if split == "test":
                config.pop(split, None)
                continue
            raise FileNotFoundError(
                f"Dataset archive is missing {split} images: {dataset_root}"
            )
        config[split] = str(image_dir.resolve())

    config["path"] = str(dataset_root.resolve())
    output_yaml = dataset_root / "coin_data.yaml"
    output_yaml.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Prepared coin dataset: {dataset_root}")
    print(f"Classes: {config['names']}")
    return output_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a local YOLO coin detector on a YOLO-format dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Dataset ZIP or data.yaml (default: coin ZIP in train/).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Initial detection weights (default: repository yolov8n.pt).",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    data_path = args.data.resolve()
    model_path = args.model.resolve()
    if not data_path.is_file():
        parser.error(f"Dataset file does not exist: {data_path}")
    if not model_path.is_file():
        parser.error(f"Initial model weights do not exist: {model_path}")
    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1 or args.workers < 0:
        parser.error(
            "epochs, imgsz, and batch must be positive; workers cannot be negative."
        )
    if args.patience < 0:
        parser.error("patience cannot be negative.")

    try:
        config_path = prepare_data_yaml(data_path)
    except (OSError, ValueError, zipfile.BadZipFile, yaml.YAMLError) as error:
        parser.error(str(error))

    model = YOLO(str(model_path))
    if model.task != "detect":
        parser.error(f"Expected detection weights, but model task is {model.task!r}.")

    model.train(
        data=str(config_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        project=str(DEFAULT_PROJECT),
        name="coin_yolov8n",
        pretrained=True,
    )


if __name__ == "__main__":
    main()
