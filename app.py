import os
from flask import Flask, current_app, jsonify, render_template, request

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

from damage_pipeline import ModelUnavailableError, analyze_image
from groq_service import GroqAPIError, estimate_repair_cost, generate_repair_report

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/damage', methods=['POST'])
def predict_damage():
    upload = request.files.get('image')
    if upload is None or not upload.filename:
        return jsonify({'success': False, 'error': '진단할 차량 이미지를 업로드해 주세요.'}), 400

    image_bytes = upload.read()
    if not image_bytes:
        return jsonify({'success': False, 'error': '업로드한 이미지가 비어 있습니다.'}), 400

    try:
        result = analyze_image(image_bytes)
    except ValueError as error:
        return jsonify({'success': False, 'error': str(error)}), 400
    except ModelUnavailableError as error:
        current_app.logger.error("AI model setup failed: %s", error)
        return jsonify({'success': False, 'error': str(error)}), 503

    try:
        result['repair_estimate'] = estimate_repair_cost(result)
    except GroqAPIError as error:
        current_app.logger.warning("Groq estimate unavailable: %s", error)
        result['repair_estimate'] = {
            'status': 'unavailable',
            'source': 'Groq',
            'error': str(error),
        }
    return jsonify(result)

@app.route('/predict/tracking', methods=['POST'])
def predict_tracking():
    return render_template('index.html')

@app.route('/api/generate-groq-report', methods=['POST'])
def generate_groq_report():
    try:
        data = request.get_json() or {}
        detections = data.get('detections', [])
        user_notes = data.get('user_notes', '')
        repair_estimate = data.get('repair_estimate', {})
        if not isinstance(detections, list) or not isinstance(repair_estimate, dict):
            return jsonify({'success': False, 'error': '손상 분석 데이터 형식이 올바르지 않습니다.'}), 400
        if not isinstance(user_notes, str) or len(user_notes) > 2000:
            return jsonify({'success': False, 'error': '추가 메모는 2,000자 이내로 입력해 주세요.'}), 400
        report = generate_repair_report(detections, repair_estimate, user_notes)
        return jsonify({
            'success': True,
            'source': f'Groq ({os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")})',
            'report': report
        })
    except GroqAPIError as error:
        current_app.logger.warning("Groq report unavailable: %s", error)
        return jsonify({'success': False, 'error': str(error)}), 503
    except Exception:
        current_app.logger.exception("Groq report processing failed.")
        return jsonify({'success': False, 'error': '리포트 처리 중 오류가 발생했습니다.'}), 500


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({
        'success': False,
        'error': '이미지 파일은 15MB 이하로 업로드해 주세요.'
    }), 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)