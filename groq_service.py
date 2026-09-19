import json
import logging
import math
import os

logger = logging.getLogger(__name__)
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "qwen/qwen3.8-27b"


class GroqAPIError(RuntimeError):
    pass


def _client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise GroqAPIError(
            "Groq API를 사용하려면 서버에 GROQ_API_KEY를 설정하세요."
        )

    try:
        from openai import OpenAI
    except ImportError as error:
        raise GroqAPIError(
            "Groq API용 OpenAI 호환 패키지가 없습니다. requirements.txt를 설치하세요."
        ) from error

    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("GROQ_BASE_URL", DEFAULT_BASE_URL),
        timeout=90.0,
    )


def _model_name():
    return os.environ.get("GROQ_MODEL", DEFAULT_MODEL)


def _completion_content(response):
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise GroqAPIError("Groq가 예상하지 못한 응답을 반환했습니다.") from error
    if not isinstance(content, str) or not content.strip():
        raise GroqAPIError("Groq가 비어 있는 응답을 반환했습니다.")
    return content.strip()


def _request_failed(error, action):
    logger.exception("Groq %s request failed.", action)
    raise GroqAPIError(
        f"Groq {action} 요청에 실패했습니다. API 키, 모델명, 서버 상태를 확인하세요."
    ) from error


def estimate_repair_cost(analysis):
    detections = analysis["detections"]
    if not detections:
        return {
            "status": "unavailable",
            "source": "Groq",
            "error": (
                "손상이 감지되지 않아 견적을 산출하지 않았습니다. "
                "손상이 있어 보이면 사진을 바꿔 다시 촬영하고 정비소에서 확인하세요."
            ),
        }

    client = _client()
    damage_summary = [
        {
            "damage_type": item["class_name_ko"],
            "detection_confidence": round(item["confidence"], 3),
            "part": item["part_name"],
            "part_classification_confidence": round(item["part_confidence"], 3),
            "mask_area_cm2": (
                round(item["area_cm2"], 2) if item["area_cm2"] is not None else None
            ),
            "mask_area_pixels": item["area_pixels"],
            "suggested_repair": item["repair"],
        }
        for item in detections
    ]
    prompt = f"""
한국 자동차 외장 수리비의 참고 견적을 작성하세요. 아래 로컬 컴퓨터 비전 결과만
검토하세요. 원본 사진은 Groq에 전달되지 않았으며, 분석 결과에 없는 내부 손상,
차종·연식·지역 정보는
불확실성으로 명시하세요. 감지 신뢰도가 낮거나 면적이 px²뿐이면 확정적으로
판단하지 말고 넓은 범위와 낮은 신뢰도를 사용하세요. 정비소의 공식 견적처럼
단정하지 말고 대한민국 원화 기준의 대략적인 참고 금액만 제시하세요. cm² 값은
탐지된 동전 지름으로 계산한 참고값이므로 실제 동전인지와 손상 부위와 같은 평면인지
확인해야 합니다. 동전이 없거나 오탐인 경우 px²만 참고하고 실제 면적으로 간주하지 마세요.

감지 결과:
{json.dumps(damage_summary, ensure_ascii=False)}

응답은 설명 없이 아래 스키마의 JSON 객체 하나만 반환하세요.
{{
  "confidence": "low 또는 medium 또는 high",
  "notes": "견적의 주요 가정과 현장 점검 필요성을 포함한 한국어 안내",
  "line_items": [
    {{
      "repair": "권장 수리 항목",
      "min_krw": 정수,
      "max_krw": 정수,
      "notes": "해당 손상별 산정 근거"
    }}
  ]
}}
line_items는 감지 손상 건수와 동일해야 합니다. 금액은 음수가 아니며 max_krw는
min_krw 이상이어야 합니다.
"""
    try:
        response = client.chat.completions.create(
            model=_model_name(),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_completion_tokens=1200,
        )
    except GroqAPIError:
        raise
    except Exception as error:
        _request_failed(error, "견적")

    try:
        result = json.loads(_completion_content(response))
        confidence = result["confidence"]
        notes = result["notes"]
        line_items = result["line_items"]
        if confidence not in {"low", "medium", "high"}:
            raise ValueError("Invalid confidence value.")
        if not isinstance(notes, str) or not isinstance(line_items, list):
            raise ValueError("Invalid response fields.")
        if len(line_items) != len(detections):
            raise ValueError("The number of estimate items does not match detections.")

        validated_items = []
        for item in line_items:
            minimum = item["min_krw"]
            maximum = item["max_krw"]
            if (
                isinstance(minimum, bool)
                or isinstance(maximum, bool)
                or not isinstance(minimum, (int, float))
                or not isinstance(maximum, (int, float))
                or not math.isfinite(minimum)
                or not math.isfinite(maximum)
                or minimum < 0
                or maximum < minimum
            ):
                raise ValueError("Invalid estimate amounts.")
            if not isinstance(item["repair"], str) or not isinstance(
                item["notes"], str
            ):
                raise ValueError("Invalid estimate descriptions.")
            validated_items.append(
                {
                    "repair": item["repair"],
                    "min_krw": round(minimum),
                    "max_krw": round(maximum),
                    "notes": item["notes"],
                }
            )
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.exception("Groq returned an invalid repair estimate.")
        raise GroqAPIError(
            "Groq가 유효한 견적 JSON을 반환하지 않았습니다. 다시 시도하세요."
        ) from error

    return {
        "status": "complete",
        "source": f"Groq ({_model_name()})",
        "total_min_krw": sum(item["min_krw"] for item in validated_items),
        "total_max_krw": sum(item["max_krw"] for item in validated_items),
        "confidence": confidence,
        "notes": notes,
        "line_items": validated_items,
    }


def generate_repair_report(detections, repair_estimate, user_notes=""):
    damage_summary = [
        {
            "damage_type": item.get("class_name_ko", "손상"),
            "part": item.get("part_name", "미분류"),
            "confidence": item.get("confidence"),
            "part_confidence": item.get("part_confidence"),
            "area_cm2": item.get("area_cm2"),
            "area_pixels": item.get("area_pixels"),
            "suggested_repair": item.get("repair"),
        }
        for item in detections
    ]
    prompt = f"""
당신은 한국 자동차 외장 수리에 관한 참고 정보를 제공하는 AI입니다. 아래 차량 손상
분석과 참고 견적을 토대로 한국어 마크다운 보고서를 작성하세요. 공식 정비 견적이나
보험 약관 안내인 것처럼 단정하지 말고, 차종·연식·숨은 손상 확인을 위해 정비소
현장 점검이 필요하다고 명시하세요. 보험 처리는 보험사와 약관에 직접 확인하도록
안내하세요.

[손상 분석]
{json.dumps(damage_summary, ensure_ascii=False)}

[수리비 참고 견적]
{json.dumps(repair_estimate, ensure_ascii=False)}

[사용자 메모]
{user_notes or "없음"}

다음 항목을 간결하게 포함하세요:
1. 손상 종합 평가
2. 부품별 권장 수리 방법
3. Groq 참고 수리비 범위와 견적의 한계
4. 정비소에서 확인할 사항과 복수 견적 권장
"""
    try:
        response = _client().chat.completions.create(
            model=_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_completion_tokens=1800,
        )
        return _completion_content(response)
    except GroqAPIError:
        raise
    except Exception as error:
        _request_failed(error, "리포트")
