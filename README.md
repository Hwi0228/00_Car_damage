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

## 로컬 동전 감지 모델 학습

`train/loaded kor coin.v2i.yolov8.zip`은 Roboflow Universe의 한국 동전 데이터셋
(CC BY 4.0, 277장, 10/50/100/500원 4개 클래스)이다. 스크립트가 ZIP을
`train/coin_dataset/`에 풀고 YOLOv8 설정 경로를 맞춘 뒤 학습한다. 로컬
`yolov8n.pt`에서 시작하며 기본 설정은 CPU, 416px, batch 8, 최대 15 epoch이고,
validation 성능이 5 epoch 연속 개선되지 않으면 일찍 멈춘다.
학습 결과는 `train/coin_runs/` 아래에 저장된다.

```powershell
python .\train\train_coin_detector.py
```

`train\damage_extent.pt`는 동전 모델이 아니라 Scratch, Dent, Breakage, Separated
4개 손상 유형의 세그멘테이션 모델이다. 손상 면적을 cm²로 환산하려면 동전 감지뿐
아니라 동전 종류(실제 지름)와 사진에서 동전/손상이 같은 평면에 놓였다는 전제도
필요하다. 차량 사진 30장 점검 중 주유구 등 원형 차량 부품을 동전으로 오인하는
사례가 확인되었지만, 실제 동전이 포함된 사진에서는 탐지된 동전의 지름으로 cm²
환산을 수행한다. 오탐이나 동전과 손상 부위의 평면 차이가 있으면 환산값이 틀릴 수
있으며, 동전을 검출하지 못하면 손상 마스크 면적을 px²로 제공한다.

## Colab 표본 학습

전체 원천 데이터를 올리는 대신, 클래스별 표본 데이터와 업로드용 ZIP을 만든다.
Bandizip CLI(`bz`)가 PATH에 있어야 한다.

```powershell
python .\train\prep.py --prepare-only --zip-for-colab
```

기본 설정은 학습 이미지 95,000장과 검증 이미지 23,000장이다. 생성한 업로드 ZIP은
약 9.38GB이며, 데이터별 이미지 용량에
따라 달라질 수 있다. 이미지와 라벨 JSON은 파일명 stem으로 짝짓고, 파손 클래스는
파일명에 포함되지 않아 선택 후보 JSON의 라벨은 8개 스레드로 읽는다. 변환 때는
이미 읽은 라벨 정보를 재사용한다. 생성되는 `train\yolo_dataset_sample_95k_colab.zip`을
Google Drive에 올린 뒤, Colab에서 ZIP을 `/content`로 복사해 압축을 풀고 로컬
런타임 디스크에서 학습한다. ZIP 안의 `data.yaml`은
`/content/yolo_dataset_sample_95k` 경로를 사용한다. 학습 결과는 런타임이 종료되기
전에 Drive로 복사한다.

Colab 학습 노트북: [train/colab_train_95k.ipynb](./train/colab_train_95k.ipynb)

## 로컬 손상 부품 분류 학습 (EfficientNet-B3)

`train\part`의 `damage_part` 데이터는 손상 마스크와 차량 부품 마스크가 별도
annotation으로 제공된다. 분류 노트북은 손상 bbox와 부품 bbox의 겹침 비율로 손상
crop의 부품 라벨을 정한 뒤 EfficientNet-B3를 학습한다. VS Code에서 저장소를 열고
로컬 Python 커널로 [train/local_part_classifier_effnetb3.ipynb](./train/local_part_classifier_effnetb3.ipynb)을
실행한다. 라벨은 앞범퍼, 앞펜더, 전조등, 뒷범퍼, 뒷펜더, 후미등, 옆면(앞/뒤 도어),
사이드미러, 스텝, 타이어/휠의 10개 그룹으로 묶는다. 데이터에 타이어 전용 라벨이
없어 타이어와 휠은 한 클래스다. 기본 로컬 학습은 train/validation 라벨 JSON을 최대
12,000개/4,000개 사용하고, 클래스당 최대 200/50 crop(최대 2,000/500장)을 생성한다.
ImageNet 사전학습 가중치, 224px 입력, batch 4로 최대 5 epoch 학습하며, validation
Macro F1이 2 epoch 연속 개선되지 않으면 조기 종료한다. 사전학습 가중치 최초 다운로드를
제외하고도 CPU 속도에 따라 수십 분 걸릴 수 있다. 체크포인트와 crop은
`train/car_damage_runs/` 아래에 저장된다. CPU에서 전체 학습 설정으로 전환하지 않는다.

## Colab 부품 분류 학습 (EfficientNet-B3)

로컬 원본 ZIP 네 개는 약 16GB라 Drive에 그대로 올리지 않는다. 전체 annotation에서
10개 부품 그룹으로 분류되는 손상 crop을 전부 생성한다(train 341,088장,
validation 60,095장). `python train/prepare_part_classifier_colab.py`를 실행하면
Bandizip이 `train/car_damage_runs/car_damage_part_crops_10gb.zip`으로 패키징하고
무결성을 검사한다. 이미 만든 `train/코랩_부품.zip`도 전체 crop 401,183장과 manifest를
포함하며 무결성 검사를 통과했다. 두 ZIP 중 하나를 서브 Google 계정 Drive의
`MyDrive/car_damage_part_classifier/`에 올리면 노트북이 자동으로 찾고 내부 폴더 구조도
처리한다. Colab 노트북
[train/colab_part_classifier_effnetb3.ipynb](./train/colab_part_classifier_effnetb3.ipynb)을
GPU 런타임에서 실행한다. 최대 5 epoch, batch 16, ImageNet 사전학습 가중치를 사용하고
validation Macro F1 기반 조기 종료 및 Drive 체크포인트 저장을 한다.

## 통합 차량 손상 분석 웹 앱

`python app.py`로 실행한 뒤 브라우저에서 `http://127.0.0.1:5000`을 연다. 차량 사진을
올리면 Flask 서버가 손상 YOLOv8-seg, 동전 YOLO, EfficientNet-B3 부품 분류를 순서대로
실행한다. 손상 마스크와 탐지 결과를 겹쳐 표시한다. 동전 모델은 후보만 표시하며,
동전이 검출되면 알려진 지름으로 마스크 면적을 cm²로 환산한다. 차량의 원형 부품을
동전으로 오인하거나 손상과 동전의 평면이 다르면 오차가 커질 수 있으니 실제 동전을
손상과 같은 평면에 두고 촬영한다. 동전이 검출되지 않으면 손상 면적은 px²로 표시한다.

수리비 학습 데이터와 별도 예측 모델은 아직 없다. 수리비 범위는 Groq API가 로컬
비전 모델의 구조화된 분석 결과를 바탕으로 생성하는 참고 견적이며, 공식 정비 견적이 아니다. 실제 가격은
차종·연식·부품·내부 손상·지역에 따라 달라진다. 견적 생성을 위해 서버 환경 변수
`GROQ_API_KEY`가 필요하다. 기본 모델은 `qwen/qwen3.8-27b`이며 `GROQ_MODEL`로 바꿀 수
있다. 사진은 로컬 YOLO 및 EfficientNet 모델에서만 처리하고, Groq에는 손상 유형·부품·
신뢰도·마스크 면적(cm² 또는 px²) 등 텍스트 분석 결과만 보낸다. 보고서 생성도 분석 결과와
참고 견적만 이용한다. 키는 채팅이나 코드에 입력하지 말고 로컬 환경 변수 또는 Render의
서비스 환경 변수에 등록한다.

2026-11-01 Groq 공개 모델 문서에서 Qwen 3.8 27B는 입력 100만 토큰당 $0.80, 출력
100만 토큰당 $4.00으로 표시되어 있어 공개 단가가 확인되는 모델 중 저렴한 선택지로
지정했다. 다만 현재 **Preview 모델**이므로 장기 운영 안정성이 보장되지 않는다.
[Groq 모델 요금 및 지원 상태](https://console.groq.com/docs/models)를 확인한다.

로컬 기본 손상 모델은 사용자가 선택한
`train/damage_extent_a14445f490c0a57759719028c4e5385634ca143a.pt`이며,
나머지 기본 모델 경로는
`train/coin_runs/coin_yolov8n-2/weights/best.pt`, `train/effb3.pt`다. `train/`의
100MB를 초과하는 체크포인트는 GitHub 파일 제한으로 Git에서 제외되므로 배포 환경에
별도로 제공하고
`DAMAGE_MODEL_PATH`, `COIN_MODEL_PATH`, `PART_MODEL_PATH`에 해당 파일 경로를 설정한다.
Render 설정에도 이 경로들이 비공개 환경 변수로 등록되어 있다. 모델이 없거나
`GROQ_API_KEY`가 설정되지 않은 경우 앱은 해당 오류를 분석 결과에 표시한다.
손상 모델은 전체 평가가 아닌 차량 사진 표본 30장, bbox IoU 0.5 기준에서 precision
19.2%, recall 2.9%를 기록했으므로 실제 손상이 있어도 누락될 수 있으며 결과를 확정 진단으로
사용해서는 안 된다.

`.gitignore`는 원본 차량/동전 데이터셋, 생성된 대량 crop 및 100MB 초과 가중치를
제외한다. 나머지 학습 코드, 작은 체크포인트, 설정, 로그 및 테스트·시각화 산출물은
Git에서 추적할 수 있다.

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
