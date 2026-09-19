import base64
import io
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
DAMAGE_NAMES_KO = {
    "Scratch": "스크래치",
    "Dent": "덴트",
    "Breakage": "파손",
    "Separated": "이격",
}
DAMAGE_COLORS = {
    "Scratch": (234, 179, 8),
    "Dent": (59, 130, 246),
    "Breakage": (239, 68, 68),
    "Separated": (168, 85, 247),
}
PART_NAMES = [
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
COIN_DIAMETERS_MM = {
    "10": 18.0,
    "50": 21.6,
    "100": 24.0,
    "500": 26.5,
}
REPAIR_SOLUTIONS = {
    "Scratch": "붓펜 및 광택 복원",
    "Dent": "PDR 덴트 복원",
    "Breakage": "부품 교체 및 용접",
    "Separated": "유격 조정 및 재조립",
}
MAX_IMAGE_PIXELS = 20_000_000
logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


def estimate_damage_area_cm2(mask_pixels, coin_diameter_pixels, coin_diameter_mm):
    if mask_pixels < 0 or coin_diameter_pixels <= 0 or coin_diameter_mm <= 0:
        raise ValueError("Pixel counts and coin diameter must be positive.")
    return mask_pixels * (coin_diameter_mm / coin_diameter_pixels) ** 2 / 100


def _estimate_area_if_coin_detected(mask_pixels, coin):
    if coin is None:
        return None
    return estimate_damage_area_cm2(
        mask_pixels, coin["diameter_pixels"], coin["diameter_mm"]
    )


def _model_path(environment_name, default_relative_path):
    configured_path = os.environ.get(environment_name)
    path = Path(configured_path) if configured_path else ROOT / default_relative_path
    return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=1)
def _load_models():
    try:
        import torch
        from torch import nn
        from torchvision import models, transforms
        from ultralytics import YOLO
    except ImportError as error:
        raise ModelUnavailableError(
            "AI inference packages are unavailable. Install requirements.txt."
        ) from error

    damage_path = _model_path(
        "DAMAGE_MODEL_PATH",
        "train/damage_extent_a14445f490c0a57759719028c4e5385634ca143a.pt",
    )
    coin_path = _model_path(
        "COIN_MODEL_PATH", "train/coin_runs/coin_yolov8n-2/weights/best.pt"
    )
    part_path = _model_path("PART_MODEL_PATH", "train/effb3.pt")
    missing_paths = [str(path) for path in (damage_path, coin_path, part_path) if not path.is_file()]
    if missing_paths:
        raise ModelUnavailableError(
            "Model weights not found: "
            + ", ".join(missing_paths)
            + ". Configure DAMAGE_MODEL_PATH, COIN_MODEL_PATH, and PART_MODEL_PATH."
        )

    damage_model = YOLO(str(damage_path))
    coin_model = YOLO(str(coin_path))
    checkpoint = torch.load(part_path, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names")
    if (
        checkpoint.get("architecture") != "efficientnet_b3"
        or class_names != PART_NAMES
    ):
        raise ModelUnavailableError(
            "The part-classifier checkpoint architecture or class order is incompatible."
        )

    part_model = models.efficientnet_b3(weights=None)
    part_model.classifier[1] = nn.Linear(
        part_model.classifier[1].in_features, len(class_names)
    )
    part_model.load_state_dict(checkpoint["model"], strict=True)
    part_model.eval()

    image_size = int(checkpoint.get("image_size", 300))
    part_transform = transforms.Compose(
        [
            transforms.Resize(320),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
    return torch, damage_model, coin_model, part_model, part_transform, class_names


def _open_image(image_bytes):
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image dimensions exceed the {MAX_IMAGE_PIXELS:,}-pixel limit."
                )
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a valid supported image.") from error


def _best_coin_detection(result):
    if result.boxes is None:
        return None

    candidates = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        raw_name = str(result.names[class_id]).strip()
        denomination = raw_name.rstrip("-")
        diameter_mm = COIN_DIAMETERS_MM.get(denomination)
        if diameter_mm is None:
            logger.warning("Ignoring unsupported coin class: %s", raw_name)
            continue
        xyxy = [float(value) for value in box.xyxy[0].tolist()]
        width = max(0.0, xyxy[2] - xyxy[0])
        height = max(0.0, xyxy[3] - xyxy[1])
        if not width or not height:
            continue
        candidates.append(
            {
                "denomination": denomination,
                "diameter_mm": diameter_mm,
                "diameter_pixels": (width + height) / 2,
                "confidence": float(box.conf.item()),
                "xyxy": xyxy,
            }
        )
    return max(candidates, key=lambda item: item["confidence"], default=None)


def _part_crop(image, box):
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    crop_width = max((x2 - x1) * 1.5, 48)
    crop_height = max((y2 - y1) * 1.5, 48)
    left = max(0, int(center_x - crop_width / 2))
    top = max(0, int(center_y - crop_height / 2))
    right = min(image.width, int(center_x + crop_width / 2 + 0.999))
    bottom = min(image.height, int(center_y + crop_height / 2 + 0.999))
    if right <= left or bottom <= top:
        raise ValueError("Damage detector returned an invalid crop.")
    return image.crop((left, top, right, bottom))


def _render_overlay(image, detections, coin):
    preview = image.copy()
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(overlay)
    for detection in detections:
        color = DAMAGE_COLORS[detection["class_name"]]
        for polygon in detection["mask_polygons"]:
            if len(polygon) >= 3:
                mask_draw.polygon(
                    [tuple(point) for point in polygon],
                    fill=(*color, 105),
                )
    preview = Image.alpha_composite(preview.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        color = DAMAGE_COLORS[detection["class_name"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{detection['class_name']} {detection['confidence']:.0%}"
        draw.text((x1, max(0, y1 - 16)), label, fill=color, font=font)

    if coin is not None:
        draw.rectangle(coin["xyxy"], outline=(245, 158, 11), width=3)
        draw.text(
            (coin["xyxy"][0], max(0, coin["xyxy"][1] - 16)),
            f"coin candidate: {coin['denomination']}",
            fill=(245, 158, 11),
            font=font,
        )

    output = io.BytesIO()
    preview.save(output, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode(
        "ascii"
    )


def analyze_image(image_bytes):
    torch, damage_model, coin_model, part_model, part_transform, part_names = (
        _load_models()
    )
    image = _open_image(image_bytes)
    damage_result = damage_model.predict(
        source=image, imgsz=640, conf=0.25, retina_masks=True, verbose=False
    )[0]
    coin_result = coin_model.predict(
        source=image, imgsz=640, conf=0.25, verbose=False
    )[0]
    coin = _best_coin_detection(coin_result)

    detections = []
    if damage_result.boxes is not None and len(damage_result.boxes) and damage_result.masks is None:
        raise RuntimeError("The damage model returned boxes without segmentation masks.")
    if damage_result.boxes is not None and damage_result.masks is not None:
        masks = damage_result.masks.data
        polygons = damage_result.masks.xy
        for index, box in enumerate(damage_result.boxes):
            class_id = int(box.cls.item())
            class_name = str(damage_result.names[class_id])
            if class_name not in REPAIR_SOLUTIONS:
                logger.warning("Ignoring unsupported damage class: %s", class_name)
                continue
            coordinates = [float(value) for value in box.xyxy[0].tolist()]
            mask = masks[index].detach().cpu().numpy() > 0.5
            if mask.shape != (image.height, image.width):
                mask_image = Image.fromarray(mask.astype("uint8") * 255)
                mask = (
                    np.array(
                        mask_image.resize(
                            (image.width, image.height), Image.Resampling.NEAREST
                        )
                    )
                    > 0
                )
            mask_pixels = int(mask.sum())
            crop = _part_crop(image, coordinates)
            with torch.inference_mode():
                probabilities = part_model(
                    part_transform(crop).unsqueeze(0)
                ).softmax(dim=1)[0]
            part_id = int(probabilities.argmax().item())
            polygon = polygons[index].tolist() if index < len(polygons) else []
            area_cm2 = _estimate_area_if_coin_detected(mask_pixels, coin)
            detections.append(
                {
                    "class_name": class_name,
                    "class_name_ko": DAMAGE_NAMES_KO[class_name],
                    "confidence": float(box.conf.item()),
                    "box": coordinates,
                    "mask_polygons": polygon,
                    "area_pixels": mask_pixels,
                    "area_cm2": area_cm2,
                    "part_name": part_names[part_id],
                    "part_confidence": float(probabilities[part_id].item()),
                    "repair": REPAIR_SOLUTIONS[class_name],
                }
            )

    annotated_image = _render_overlay(image, detections, coin)
    for detection in detections:
        detection.pop("box")
        detection.pop("mask_polygons")

    summary = {
        "damage_count": len(detections),
        "damage_types": sorted(
            {item["class_name_ko"] for item in detections}
        ),
        "total_area_pixels": sum(item["area_pixels"] for item in detections),
        "total_area_cm2": (
            sum(item["area_cm2"] for item in detections)
            if coin is not None
            else None
        ),
    }
    coin_summary = (
        {
            "detected": True,
            "candidate_detected": True,
            "denomination": coin["denomination"],
            "diameter_mm": coin["diameter_mm"],
            "confidence": coin["confidence"],
            "note": (
                "탐지된 동전의 지름으로 면적을 환산한 참고값입니다. 실제 동전인지 확인하고 "
                "손상 부위와 같은 평면에 놓였는지 확인하세요. 주유구·휠 등 오탐이면 "
                "환산값도 부정확합니다."
            ),
        }
        if coin is not None
        else {
            "detected": False,
            "candidate_detected": False,
            "note": "동전을 검출하지 못해 면적을 px²로 표시합니다. 손상 부위와 같은 평면에 둔 동전을 촬영하세요.",
        }
    )
    return {
        "success": True,
        "annotated_image": annotated_image,
        "detections": detections,
        "reference": coin_summary,
        "summary": summary,
        "estimate_type": "Groq 참고 견적",
    }
