from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', damage_result=None)

@app.route('/predict/damage', methods=['POST'])
def predict_damage():
    file = request.files.get('file')
    if file:
        result_data = { ##### 여기에 나중에 api 돌려서 나온 거 출력
            'output_image': 'https://placehold.co/600x400?text=Test%20Image',
            'part': '운전석 도어',
            'type': '문콕 / 스크래치',
            'repair': 'PDR (덴트 복원) 및 붓펜',
            'cost': '약 50,000원 ~ 100,000원'
        }
        return render_template('index.html', damage_result=result_data)
    
    return render_template('index.html', damage_result=None)

@app.route('/predict/tracking', methods=['POST'])
def predict_tracking():
    # 추적 기능 추론 로직
    return render_template('index.html')
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)