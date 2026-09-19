# 문제 정의서

아빠는 **교통사고가 발생해 차량에 손상을 입어도 어떻게 수리해야 하는지, 누구에게 책임을 져야 하는지를 모르기 때문**에 문제점을 겪고 있으며, 이를 **세그멘테이션을 사용한 수리 방법 제안과 객체 추적을 사용한 용의자 추적**으로 해결하고자 한다.

## 객체 추적

문콕 및 스크래치가 발생했을 때, 범인이 누군지를 차량의 블랙박스 영상을 넣으면 추적한다.

## 세그멘테이션

차량의 사진을 찍으면 세그멘테이션을 통해 어떤 부위가 어떻게 손상되었는지 알려주고, 어떻게 수리하면 좋을지도 알려준다.

## 워크플로우

![워크플로우](image_01_workflow_diagram.png)

## 폴더 구조

```text
00_Car_damage/
  static/             # 업로드 파일 및 결과 이미지/영상 저장
  templates/          # HTML 템플릿
    index.html        # 진단 및 추적 기능을 탭으로 통합한 단일 화면
  weights/            # AI 모델 가중치 (.pt)
    yolo_seg.pt
    yolo_track.pt
  app.py              # Flask 앱 메인 (라우팅 + AI 모델 추론 로직)
  requirements.txt    # 의존성 패키지 목록
```

## 개발 환경

GitHub: https://github.com/Hwi0228/00_Car_damage

## 페르소나

| 구분 | 값 |
|---|---|
| 나이 | 23세 |
| 상황 | 사회 초년생임. 차를 처음 구매했는데 사고가 났고, 어떻게 해결해야하는지 모르는 상태임. |
| 목표 | 어떻게 해결해야 하고 얼마나 들지, 또 용의자는 누구인지 알려줌. |
| 성공 기준 | 전문가가 예측한, 권장한 수리 방법과 얼마나 차이가 나는지 비교해 유사도 70% 이상 |

# 웹 서버

- flask를 사용할 예정 - gradia는 너무 확장성이 적음, jinja 템플릿은 LLM에게(tailwind css)
- 서버는 ~~PythonAnywhere? Render?~~ ⇒ ~~Hugging Face with Flask~~ ⇒ CPU Instance가 너무 적음, Render
- https://zero0-car-damage.onrender.com/
  - Render에서 flask 쓰는법
    - `requirements.txt` 꼭 쓰기!
    - ip를 0.0.0.0:5000으로
- client-side ai로

  ```sh
  2026-09-12T01:22:34.885575534Z Installing collected packages: MarkupSafe, itsdangerous, click, blinker, Werkzeug, Jinja2, flask
  2026-09-12T01:22:35.239523353Z
  2026-09-12T01:22:35.24071007Z Successfully installed Jinja2-3.1.6 MarkupSafe-3.0.3 Werkzeug-3.1.8 blinker-1.9.0 click-8.5.0 flask-3.0.3 itsdangerous-2.2.0
  2026-09-12T01:22:35.243960841Z
  2026-09-12T01:22:35.243975239Z [notice] A new release of pip is available: 25.3 -> 26.2.1
  2026-09-12T01:22:35.243977494Z [notice] To update, run: pip install --upgrade pip
  2026-09-12T01:22:36.761247635Z ==> Uploading build...
  2026-09-12T01:22:39.40274678Z ==> Uploaded in 1.5s. Compression took 1.1s
  2026-09-12T01:22:39.405603497Z ==> Build successful 🎉
  2026-09-12T01:22:41.185542982Z ==> Deploying...
  2026-09-12T01:22:41.380671573Z ==> Setting WEB_CONCURRENCY=1 by default, based on available CPUs in the instance
  2026-09-12T01:22:47.571798408Z ==> Running 'python app.py'
  2026-09-12T01:22:52.471569178Z  * Serving Flask app 'app'
  2026-09-12T01:22:52.47159465Z  * Debug mode: off
  2026-09-12T01:22:52.56525065Z WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
  2026-09-12T01:22:52.565266401Z  * Running on all addresses (0.0.0.0)
  2026-09-12T01:22:52.565270531Z  * Running on http://127.0.0.1:10000
  2026-09-12T01:22:52.565274551Z  * Running on http://10.25.166.130:10000
  2026-09-12T01:22:52.819232667Z 127.0.0.1 - - [12/Sep/2026 01:22:52] "HEAD / HTTP/1.1" 200 -
  2026-09-12T01:23:02.29840142Z 127.0.0.1 - - [12/Sep/2026 01:23:02] "GET / HTTP/1.1" 200 -
  2026-09-12T01:23:02.306149389Z ==> Your service is live 🎉
  2026-09-12T01:23:02.517628782Z ==>
  2026-09-12T01:23:02.522291534Z ==>
  2026-09-12T01:23:02.524697412Z ==>
  2026-09-12T01:23:02.527372767Z ==> Available at your primary URL https://zero0-car-damage.onrender.com
  2026-09-12T01:23:02.530279127Z ==>
  2026-09-12T01:23:02.533062084Z ==> ///////////////////////////////////////////////////////////
  2026-09-12T01:23:11.434095984Z 127.0.0.1 - - [12/Sep/2026 01:23:11] "GET / HTTP/1.1" 200 -
  2026-09-12T01:23:12.419978892Z 127.0.0.1 - - [12/Sep/2026 01:23:12Z] "GET /favicon.ico HTTP/1.1" 404 -
  ```

## UI 구상

![UI 구상](image_02_ui_document.png)

# 세그멘테이션

- https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=581
- 비율만 있으면 절대적인 파손 사이즈를 확인하기 어려움
  - ⇒ 누구나 똑같은 크기로 하나 씩은 가지고 있는 물건이 필요함
  - ⇒ 100원 동전, 카드 등
  - 동전 데이터:
    - https://universe.roboflow.com/visitor/loaded-kor-coin/dataset/2

# 객체 추적

- ⇒ 결국 YOLO 모델에 이동평균 한 것임.
- ⇒ 그러면 학습을 해야 하는데, 사람은 이미 되어 있음.

# 테스트 계획

- 나 말고 다른 사람이 아파트 주차장에서 스크래치 있는 차 확인해서 사진 찍어보기
- 객체 추적은 아빠 차의 블랙박스로 확인하기.

# 제약 검토

- 세그멘테이션
  - 사진이 잘 찍히지 않을 수 있음. 잘릴 수 있음 등등
- 객체 추적
  - 어두운 밤 등에는 추적이 불가능 할 수 있음.
