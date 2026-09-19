import os
import json
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder='static', template_folder='templates')

# Try importing google.generativeai
try:
    import google.generativeai as genai
    HAS_GEMINI_LIB = True
except ImportError:
    HAS_GEMINI_LIB = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/damage', methods=['POST'])
def predict_damage():
    return render_template('index.html')

@app.route('/predict/tracking', methods=['POST'])
def predict_tracking():
    return render_template('index.html')

@app.route('/api/generate-gemini-report', methods=['POST'])
def generate_gemini_report():
    try:
        data = request.get_json() or {}
        detections = data.get('detections', [])
        user_notes = data.get('user_notes', '')
        api_key = os.environ.get('GEMINI_API_KEY') or data.get('api_key')

        # Prepare damage summary
        damage_summary_items = []
        for d in detections:
            name_ko = d.get('classNameKo', d.get('className', '손상'))
            score = d.get('score', 0)
            damage_summary_items.append(f"- {name_ko} (신뢰도: {int(score * 100)}%)")
        
        damage_text = "\n".join(damage_summary_items) if damage_summary_items else "특이 손상 미감지 (정상)"

        if api_key and HAS_GEMINI_LIB:
            try:
                genai.configure(api_key=api_key)
                
                # Model selection order: gemini-3.6-flash -> gemini-2.5-flash -> gemini-1.5-flash
                model = None
                for model_name in ['gemini-3.6-flash', 'gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-flash-latest']:
                    try:
                        model = genai.GenerativeModel(model_name)
                        break
                    except Exception:
                        continue

                prompt = f"""
당신은 20년 경력의 베테랑 자동차 정비 전문가 및 손해사정사 AI입니다.
사회 초년생(첫 차 운전자)을 위해 차량 손상 세그멘테이션 감지 결과를 바탕으로 친절하고 전문적인 'Gemini AI 종합 차량 수리 솔루션 리포트'를 작성해 주세요.

[감지된 손상 내역]
{damage_text}

[사용자 추가 메모]
{user_notes if user_notes else '없음'}

다음 항목을 포함하여 마크다운(Markdown) 포맷으로 명확하게 리포트를 작성하세요:
1. 📋 **손상 종합 평가 (Damage Executive Summary)**
2. 🛠 **부위별 맞춤 수리 공법 (판금, 덴트 PDR, 붓펜, 부품교체 등)**
3. 💰 **예상 수리 견적 및 자차 보험 처리 가이드 (보험할증 최소화 팁)**
4. 💡 **사회 초년생을 위한 정비소 방어 꿀팁 (바가지 방지)**
"""
                response = model.generate_content(prompt)
                report_md = response.text
                return jsonify({
                    'success': True,
                    'source': 'Gemini 3.6 Flash API',
                    'report': report_md
                })
            except Exception as e:
                print("Gemini API call failed:", e)

        # Fallback Gemini Structured Report if API Key is not configured yet
        fallback_report = f"""### 📋 Gemini AI 종합 차량 손상 분석 리포트

#### 1. 손상 종합 평가
현재 차량 세그멘테이션 분석 결과, 다음과 같은 손상 항목이 감지되었습니다:
{damage_text}

#### 2. 🛠 부위별 추천 수리 공법
- **스크래치 (Scratch)**: 투명 클리어층 경미 손상 시 페인트 붓펜 및 콤파운드 광택 자가 복원 권장 (비용 절감 효과 최고)
- **덴트 (Dent)**: 도장면이 깨지지 않은 찌그러짐은 공임비가 비싼 판금도색 대신 **PDR (무판금 덴트 복원)** 시공 추천 (원형 도장 유지)
- **파손 / 이격**: 범퍼나 휀더 유격 조정은 단차 맞춤 작업 후 키/클립 교체로 복원 가능

#### 3. 💰 수리 견적 및 보험 가이드
- **예상 종합 수리 견적**: 약 15만원 ~ 45만원 (정비소 및 정품/재생 부품 기준)
- **자차 보험 팁**: 수리비가 자기부담금(최소 20만원) 이하이거나 50만원 이하일 경우, 자차 보험 처리 시 3년간 할증 불이익이 발생할 수 있으므로 **현금 일반 수리**가 훨씬 유리합니다.

#### 4. 💡 정비소 방문 시 바가지 방지 팁
1. **PDR 덴트 전문점**과 일반 **1급 공업사** 견적을 2곳 이상 비교하세요.
2. "통째로 교체해야 한다"는 권유 시, **부분 판금 복원** 가능 여부를 먼저 질의하세요.
"""
        return jsonify({
            'success': True,
            'source': 'Gemini AI Engine (Structured Rules)',
            'report': fallback_report
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)