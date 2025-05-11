
# Pet-i Behavior Classification

## 개요
이 프로젝트는 자이로스코프와 가속도계 센서 데이터를 사용하여 반려견의 6가지 행동 상태(걷기, 뛰기, 쉬기 등)를 예측하는 딥러닝 모델을 개발하는 것을 목표로 합니다.  
초기 버전은 완전연결 신경망(MLP)을 사용하며, 추후 CNN+LSTM 구조로 모델을 고도화할 계획입니다.

## 주요 기능
- **데이터 로드 & 전처리**: CSV 파일로 제공되는 센서 데이터를 불러와 특성 선택, 스케일링, 레이블 인코딩 수행
- **모델 정의**: 간단한 MLP 구조의 신경망 모델 생성 및 컴파일
- **모델 훈련 & 평가**: 훈련/검증 분할, 에포크별 학습 진행 및 테스트 정확도 평가
- **예측 & 시각화**: 학습된 모델로 샘플 예측 수행 및 손실·정확도 학습 곡선 시각화

## 프로젝트 구조
```
.
├── dataset/
│   ├── DogInfo.csv         # 강아지 메타 정보 (ID, 품종 등)
│   └── DogMoveData.csv     # 센서 데이터와 행동 라벨
├── Pet\_i\_Behavior\_Classification.ipynb  # 모듈화된 Jupyter Notebook
├── main.py                 # 모듈화 이전의 스크립트 버전 (참고용)
├── README.md               # 프로젝트 설명서
└── requirements.txt        # 프로젝트 의존성 목록

````

## 요구 사항 (Requirements)
- Python 3.8+
- MacOS: TensorFlow 2.6+
- scikit-learn 1.x  
- pandas 1.x  
- numpy 1.x  
- matplotlib 3.x  

```bash
# 의존성 설치 (requirements.txt 파일 활용)
pip install -r requirements.txt
````

## 설치 및 실행 (Installation & Usage)

1. 이 저장소를 클론/clojure:

   ```bash
   git clone https://github.com/dpeyvc/Pet-i_Behavior_Classfication.git
   cd Pet-i_Behavior_Classfication
   ```
2. 가상환경(선택) 생성 및 활성화:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. 필요한 패키지 설치:

   ```bash
   pip install -r requirements.txt
   ```
4. `dataset/` 폴더에 `DogInfo.csv`와 `DogMoveData.csv` 파일을 위치시킵니다 ([Kaggle Dataset](https://www.kaggle.com/datasets/benjamingray44/inertial-data-for-dog-behaviour-classification)).
5. Jupyter Notebook 실행:

   ```bash
   jupyter notebook Pet_i_Behavior_Classification.ipynb
   ```
6. 또는 스크립트 직접 실행:

   ```bash
   python main.py
   ```

## 사용 예시 (Example)

노트북 또는 스크립트를 실행하면 다음과 같은 출력 및 파일이 생성됩니다:

```
Train samples: (41440, 12), Test samples: (10360, 12)
Epoch 1/35
  1295/1295 - loss: 1.2345 - accuracy: 0.5678 - val_loss: 1.1456 - val_accuracy: 0.5901
...
Test samples: Loss=0.9876, Accuracy=0.6234
샘플 0 예측 클래스: 걷기
```

학습 곡선 플롯이 자동으로 표시되며, 모델 파일(`pet_activity_model.h5`)로 저장합니다.
