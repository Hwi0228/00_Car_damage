import io
import json
import os
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import app as app_module
from damage_pipeline import _estimate_area_if_coin_detected, estimate_damage_area_cm2
from groq_service import GroqAPIError, estimate_repair_cost


def image_bytes():
    image = Image.new("RGB", (32, 32), color="white")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


class DamageAreaTests(unittest.TestCase):
    def test_area_uses_coin_scale(self):
        area = estimate_damage_area_cm2(100, 20, 20)
        self.assertAlmostEqual(area, 1.0)

    def test_area_rejects_invalid_coin_measurement(self):
        with self.assertRaises(ValueError):
            estimate_damage_area_cm2(100, 0, 20)

    def test_analysis_area_is_scaled_only_when_coin_is_detected(self):
        coin = {"diameter_pixels": 20, "diameter_mm": 20}

        self.assertAlmostEqual(_estimate_area_if_coin_detected(100, coin), 1.0)
        self.assertIsNone(_estimate_area_if_coin_detected(100, None))


class GroqEstimateTests(unittest.TestCase):
    def test_estimate_uses_validated_line_items(self):
        response_data = {
            "confidence": "medium",
            "notes": "현장 점검이 필요합니다.",
            "line_items": [
                {
                    "repair": "판금도색",
                    "min_krw": 120000,
                    "max_krw": 240000,
                    "notes": "외관 손상 기준입니다.",
                }
            ],
        }
        create_completion = Mock(
            return_value=types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(response_data))
                    )
                ]
            )
        )
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=create_completion
                )
            )
        )
        analysis = {
            "detections": [
                {
                    "class_name_ko": "덴트",
                    "confidence": 0.9,
                    "part_name": "앞범퍼",
                    "part_confidence": 0.8,
                    "area_cm2": 1.0,
                    "area_pixels": 100,
                    "repair": "PDR 덴트 복원",
                }
            ],
            "reference": {
                "candidate_detected": True,
                "denomination": "100",
                "note": "검증되지 않은 동전 후보",
            },
        }
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            patch("groq_service._client", return_value=fake_client),
        ):
            result = estimate_repair_cost(analysis)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["total_min_krw"], 120000)
        self.assertEqual(result["total_max_krw"], 240000)
        self.assertEqual(result["line_items"][0]["repair"], "판금도색")
        self.assertIn("Groq", result["source"])
        request = create_completion.call_args.kwargs
        self.assertEqual(request["model"], "qwen/qwen3.8-27b")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertIsInstance(request["messages"][0]["content"], str)
        self.assertNotIn("image_url", request["messages"][0])
        self.assertIn('"damage_type": "덴트"', request["messages"][0]["content"])
        self.assertIn('"mask_area_cm2": 1.0', request["messages"][0]["content"])
        self.assertNotIn('"denomination": "100"', request["messages"][0]["content"])

    def test_missing_detections_return_unavailable_without_api_call(self):
        with patch("groq_service._client") as client:
            result = estimate_repair_cost({"detections": [], "reference": {}})

        self.assertEqual(result["status"], "unavailable")
        client.assert_not_called()

    def test_missing_api_key_is_reported(self):
        analysis = {"detections": [{"class_name_ko": "스크래치"}], "reference": {}}
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GroqAPIError, "GROQ_API_KEY"):
                estimate_repair_cost(analysis)


class DamageRouteTests(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_predict_damage_returns_analysis_and_estimate(self):
        analysis = {
            "success": True,
            "annotated_image": "data:image/jpeg;base64,",
            "detections": [],
            "reference": {"detected": False, "note": "동전 없음"},
            "summary": {
                "damage_count": 0,
                "damage_types": [],
                "total_area_pixels": 0,
                "total_area_cm2": None,
            },
        }
        estimate = {
            "status": "complete",
            "source": "Groq",
            "total_min_krw": 0,
            "total_max_krw": 0,
            "line_items": [],
        }
        with (
            patch("app.analyze_image", return_value=analysis),
            patch("app.estimate_repair_cost", return_value=estimate) as groq_estimate,
        ):
            response = self.client.post(
                "/predict/damage",
                data={"image": (io.BytesIO(image_bytes()), "car.jpg")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["summary"]["damage_count"], 0)
        self.assertEqual(response.json["repair_estimate"]["source"], "Groq")
        groq_estimate.assert_called_once_with(analysis)

    def test_missing_image_is_rejected(self):
        response = self.client.post("/predict/damage")
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["success"])

    def test_groq_report_uses_groq_service(self):
        with patch("app.generate_repair_report", return_value="### 수리 안내") as report:
            response = self.client.post(
                "/api/generate-groq-report",
                json={
                    "detections": [{"class_name_ko": "덴트"}],
                    "repair_estimate": {"status": "complete"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["source"], "Groq (qwen/qwen3.8-27b)")
        report.assert_called_once()

    def test_groq_report_rejects_non_string_user_notes(self):
        response = self.client.post(
            "/api/generate-groq-report",
            json={"user_notes": ["unexpected"]},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
