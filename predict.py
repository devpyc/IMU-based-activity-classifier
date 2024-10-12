import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

# 저장된 모델 불러오기
model = load_model('wandb_model_final.h5')

# 입력 데이터 (자이로스코프 x, y, z 및 가속도계 x, y, z 순서)
input_data = np.array([[-10, 10, -10, -2, 2, 0]])  # 예시 데이터

# 입력 데이터 정규화 (훈련 시 사용한 범위인 -1에서 1로 스케일링)
# 기존 훈련 시 사용한 MinMaxScaler를 불러와야 하므로, 훈련 시 저장했던 scaler를 불러오는 것이 이상적입니다.
# 아래는 scaler를 새로 생성하여 사용하는 예시입니다. (실제 사용 시에는 훈련 시 사용한 scaler를 저장하고 불러오는 것이 좋습니다.)
scaler = MinMaxScaler(feature_range=(-1, 1))
input_data_normalized = scaler.fit_transform(input_data.reshape(-1, 1)).reshape(1, -1)

# 입력 데이터 전처리 (CNN 모델을 위해 3차원 배열로 변환)
input_data_reshaped = np.expand_dims(input_data_normalized, axis=-1)  # (1, 6, 1)

# 모델로 예측
prediction = model.predict(input_data_reshaped)

# 예측 결과 해석 (가장 높은 확률을 가진 클래스 선택)
predicted_class = np.argmax(prediction, axis=-1)

# 라벨 인코더 불러오기 및 활동 상태 정의
activity_data = ['Walking', 'Trotting', 'Sitting', 'Standing', 'Shaking', 'Galloping', 'Lying chest']  # 7개 활동 상태

# 라벨 인코더에 활동 상태 학습시키기
le = LabelEncoder()
le.fit(activity_data)

# 예측된 클래스 출력
predicted_activity = le.inverse_transform(predicted_class)
print(f"예측된 활동 상태: {predicted_activity[0]}")