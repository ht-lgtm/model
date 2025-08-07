#YOLOv11n 기반 불법 주정차 탐지 시스템
🚗 프로젝트 개요
본 프로젝트는 YOLOv11n (You Only Look Once version 11 nano) 모델을 활용하여 CCTV 영상에서 불법 주정차 차량을 실시간으로 탐지하는 시스템을 개발하는 것을 목표로 합니다. AI-Hub의 '종합민원 이미지' 데이터셋과 ITS 국가교통정보센터에서 제공하는 실제 교통 영상 데이터를 사용하여 모델을 학습하고 성능을 검증했습니다.

⚙️ 개발 과정
프로젝트는 크게 데이터 전처리, 모델 학습, 추론 및 검증의 세 단계로 진행되었습니다.

1. 데이터 전처리 (app.py)
목표: YOLO 모델 학습에 적합한 형태로 원시 데이터를 가공합니다.
데이터 출처: AI-Hub 종합민원 이미지 데이터
내용:

학습에 사용할 영상 데이터에서 프레임 단위로 이미지를 추출합니다.

추출된 이미지에 대해 Labeling 작업을 진행하여, 탐지할 객체(자동차, 버스, 트럭 등)의 위치를 지정하는 Bounding Box와 클래스 정보를 담은 라벨 파일을 생성합니다.

app.py 스크립트를 사용하여 이 과정을 자동화하고, 학습 데이터셋과 검증 데이터셋을 구축합니다.

2. YOLOv11n 모델 학습 (yolo_test.ipynb)
목표: 전처리된 데이터를 사용하여 불법 주정차 차량 탐지 모델을 학습시킵니다.
내용:

경량화된 객체 탐지 모델인 YOLOv11n을 기반으로 선택하여 빠른 추론 속도와 준수한 성능을 확보하고자 했습니다.

제공된 yolo_test.ipynb 파일의 코드를 기반으로 Google Colab의 GPU 환경을 활용하여 학습을 진행했습니다.

전처리된 데이터셋을 이용해 모델을 Fine-tuning(미세 조정)하여, 우리 데이터에 특화된 객체 탐지 모델(best.pt 가중치 파일)을 생성했습니다.

학습 과정 전체 코드 및 결과:
1단계: 필요 라이브러리 설치

# YOLO 모델을 사용하기 위해 필요한 ultralytics 라이브러리를 설치합니다.
!pip install ultralytics


실행 결과:

Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt to 'yolov8n.pt': 100% 6.25M/6.25M [00:00<00:00, 63.7MB/s]
...
Successfully installed ultralytics-8.3.175


2단계: Google Drive 연동

# 데이터셋이 저장된 Google Drive에 접근하기 위해 드라이브를 마운트합니다.
from google.colab import drive
drive.mount('/content/drive')


실행 결과:

Mounted at /content/drive


3단계: YOLOv11n 모델 로드 및 학습

from ultralytics import YOLO

# 사전 학습된 yolov11n 모델을 로드합니다.
model = YOLO('yolov11n.pt')

# data.yaml 파일에 정의된 경로의 데이터셋으로 모델을 학습시킵니다.
# epochs: 전체 데이터셋 반복 횟수
# imgsz: 학습에 사용될 이미지 크기
results = model.train(data='/content/datasets/data.yaml', epochs=300, imgsz=640)


실행 결과:

Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% 755k/755k [00:00<00:00, 13.3MB/s]
...
Overriding model.yaml nc=80 with nc=2
...
Model summary: 129 layers, 3,011,238 parameters, 3,011,222 gradients, 8.2 GFLOPs
...
Starting training for 300 epochs...
...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/300      2.04G      1.163      3.087      1.145         54        640: 100% 44/44 [00:08<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 7/7 [00:02<00:00,  3.24it/s]
                   all        200        399    0.00631      0.949      0.124     0.0864

... (학습 진행) ...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/300      3.73G     0.7214     0.8989     0.9532         36        640: 100% 44/44 [00:06<00:00,  6.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 7/7 [00:01<00:00,  6.37it/s]
                   all        200        399      0.391      0.446      0.379      0.305

... (학습 진행) ...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    188/300       5.7G     0.5461     0.5062     0.8915         37        640: 100% 44/44 [00:06<00:00,  6.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 7/7 [00:01<00:00,  6.68it/s]
                   all        200        399      0.428      0.374      0.355      0.301
EarlyStopping: Training stopped early as no improvement observed in last 100 epochs. Best results observed at epoch 88, best model saved as best.pt.

188 epochs completed in 0.422 hours.
Optimizer stripped from runs/detect/train/weights/last.pt, 6.3MB
Optimizer stripped from runs/detect/train/weights/best.pt, 6.3MB

Validating runs/detect/train/weights/best.pt...
Ultralytics 8.3.175 🚀 Python-3.11.13 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 14914MiB)
Model summary (fused): 72 layers, 3,006,038 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 7/7 [00:02<00:00,  3.31it/s]
                   all        200        399      0.421      0.442      0.418      0.337
                   car        100        222        0.4      0.414      0.366      0.292
                   SUV        100        177      0.442      0.469      0.469      0.382
Speed: 0.2ms preprocess, 2.1ms inference, 0.0ms loss, 2.1ms postprocess per image
Results saved to runs/detect/train


4단계: 학습된 모델로 추론(예측) 수행

from ultralytics import YOLO

# 학습으로 생성된 가중치 파일 중 가장 성능이 좋은 best.pt를 로드합니다.
model = YOLO('/content/runs/detect/train/weights/best.pt')

# 테스트 비디오 파일을 사용하여 추론을 실행하고, 결과를 저장합니다.
results = model.predict(source='/content/drive/MyDrive/Colab Notebooks/test_video.mp4', save=True)


실행 결과:

(학습 완료 후 추론을 진행하여 최종 추론 결과를 확인할 수 있습니다.)
video 1/1 (1/2398) /content/drive/MyDrive/Colab Notebooks/test_video.mp4: 384x640 5 cars, 1 bus, 7.0ms
...
Speed: 0.2ms preprocess, 2.1ms inference, 0.0ms loss, 2.1ms postprocess per image
Results saved to runs/detect/predict


3. 추론 및 결과 확인 (my_model.py)
목표: 학습된 커스텀 모델을 실제 영상에 적용하여 성능을 검증합니다.
내용:

my_model.py 스크립트는 학습 완료된 가중치 파일(best.pt)을 불러옵니다.

ITS 국가교통정보센터에서 수집한 실제 주행 및 주차 환경이 담긴 테스트 영상을 입력합니다.

영상의 각 프레임마다 객체 탐지를 수행하여 차량의 위치를 실시간으로 추적합니다.

탐지된 차량이 특정 영역에서 일정 시간 이상 머무를 경우 '불법 주정차'로 판단하고, 해당 차량을 시각적으로 표시(예: Bounding Box 색상 변경)하여 결과를 확인합니다.

🚀 실행 방법
저장소 복제

git clone https://github.com/ht-lgtm/model.git
cd model


필요 라이브러리 설치

pip install -r requirements.txt


추론 스크립트 실행
학습된 모델 가중치(best.pt)와 테스트 영상을 사용하여 아래 명령어를 실행합니다.

python my_model.py --weights {your_model_weights.pt} --source {your_video.mp4}


{your_model_weights.pt}: 학습된 모델의 가중치 파일 경로

{your_video.mp4}: 테스트할 영상 파일 경로

✨ 결과 예시
아래는 학습된 모델이 실제 영상에서 불법 주정차 차량을 탐지하는 예시입니다.
(여기에 결과 GIF나 스크린샷을 추가하시면 좋습니다.)
