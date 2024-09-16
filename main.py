import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Metal 가속 활성화
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Metal acceleration enabled")

# 데이터 로드
dog_info = pd.read_csv('./dataset/DogInfo.csv')
dog_move_data = pd.read_csv('./dataset/DogMoveData.csv')

# 센서 데이터 선택
sensor_columns = ['ABack_x', 'ABack_y', 'ABack_z', 'ANeck_x', 'ANeck_y', 'ANeck_z',
                  'GBack_x', 'GBack_y', 'GBack_z', 'GNeck_x', 'GNeck_y', 'GNeck_z']

X = dog_move_data[sensor_columns]
y = dog_move_data['Behavior_1']  # 주요 행동만 예측

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 훈련 및 테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 고유한 행동 클래스 수 확인
num_classes = y.nunique()

# 모델 구축
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(12,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 레이블 인코딩
y_train_encoded = pd.Categorical(y_train).codes
y_test_encoded = pd.Categorical(y_test).codes

# 모델 훈련
history = model.fit(X_train, y_train_encoded,
                    epochs=35,
                    batch_size=64,
                    validation_split=0.2)

# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f'Test accuracy: {test_acc}')

# 모델 저장
model.save('gpu_test_model.h5')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 새로운 데이터에 대한 예측 (예시)
new_data = X_test[:5]  # 테스트 데이터의 처음 5개 샘플 사용
predictions = model.predict(new_data)
predicted_classes = np.argmax(predictions, axis=1)

# 예측 결과 출력
for i, pred in enumerate(predicted_classes):
    print(f"Sample {i+1} predicted behavior: {y.cat.categories[pred]}")
