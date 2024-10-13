import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, GRU
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# 2. 데이터 불러오기
data = pd.read_csv('./dataset/DogMoveData.csv')

# 3. 제거할 행동 리스트
remove_behaviors = ['Bowing', 'Jumping', 'Tugging', 'Synchronization', 'Extra_Synchronization',
                    'Sniffing', 'Playing', 'Panting', 'Eating', 'Pacing', 'Drinking', 'Carrying object', '<undefined>']

# 4. 데이터 필터링
filtered_movedata = data[~data['Behavior_1'].isin(remove_behaviors)]

# 5. Task별로 데이터 필터링
filter_conditions = (
        (filtered_movedata['Behavior_1'] == 'Walking') & (filtered_movedata['Task'] == 'Task walk') |
        (filtered_movedata['Behavior_1'] == 'Trotting') & (filtered_movedata['Task'] == 'Task trot') |
        (filtered_movedata['Behavior_1'] == 'Sitting') & (filtered_movedata['Task'] == 'Task sit') |
        (filtered_movedata['Behavior_1'] == 'Standing') & (filtered_movedata['Task'] == 'Task stand') |
        (filtered_movedata['Behavior_1'] == 'Lying chest') & (filtered_movedata['Task'] == 'Task lie down') |
        (filtered_movedata['Behavior_1'] == 'Shaking') |
        (filtered_movedata['Behavior_1'] == 'Galloping')
)

filtered_movedata = filtered_movedata[filter_conditions]

# 6. 행동별 샘플 개수 조정 및 데이터 증강
desired_count = 100000
sampled_data = []
le = LabelEncoder()
le.fit(filtered_movedata['Behavior_1'])

# 센서 열 이름을 리스트로 정의
sensor_columns = ['GNeck_x', 'GNeck_y', 'GNeck_z', 'ANeck_x', 'ANeck_y', 'ANeck_z']

for behavior in filtered_movedata['Behavior_1'].value_counts().index:
    class_data = filtered_movedata[filtered_movedata['Behavior_1'] == behavior]
    if len(class_data) > desired_count:
        sampled_data.append(class_data.sample(n=desired_count, random_state=42))
    else:
        insufficient_data = class_data.copy()
        if behavior == 'Galloping':
            while insufficient_data.shape[0] < 50000:
                noise = np.random.normal(0, 0.05, insufficient_data[sensor_columns].shape)
                augmented_data = insufficient_data.copy()
                augmented_data[sensor_columns] += noise
                insufficient_data = pd.concat([insufficient_data, augmented_data])
            sampled_data.append(insufficient_data.sample(n=desired_count, random_state=42, replace=True))
        elif behavior == 'Shaking':
            while insufficient_data.shape[0] < desired_count:
                noise = np.random.normal(0, 0.05, insufficient_data[sensor_columns].shape)
                augmented_data = insufficient_data.copy()
                augmented_data[sensor_columns] += noise
                insufficient_data = pd.concat([insufficient_data, augmented_data])
            sampled_data.append(insufficient_data)

final_data = pd.concat(sampled_data)

# 데이터 전처리
activity_column = 'Behavior_1'
sensor_data = final_data[sensor_columns].copy()
activity_data = final_data[activity_column].copy()
sensor_data.fillna(0, inplace=True)
activity_data_encoded = le.transform(activity_data)

scaler = MinMaxScaler(feature_range=(-1, 1))
sensor_data[sensor_columns] = scaler.fit_transform(sensor_data[sensor_columns])

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(sensor_data, activity_data_encoded, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# 클래스 가중치 조정
class_weights = {
    0: 1.0,   # Galloping
    1: 3.0,   # Lying chest
    2: 1.0,   # Shaking
    3: 2.0,   # Sitting
    4: 2.5,   # Standing
    5: 2.0,   # Trotting
    6: 2.0    # Walking
}

# 모델 정의
model = Sequential()
model.add(Conv1D(64, 2, activation='selu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(GRU(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

# 옵티마이저 설정
optimizer = tf.keras.optimizers.Adam()

# 7. 사용자 정의 훈련 루프
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')

    # 훈련 데이터
    for i in range(0, len(X_train), batch_size):
        # 배치 데이터 선택
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        with tf.GradientTape() as tape:
            # 모델 예측
            predictions = model(tf.expand_dims(X_batch, axis=-1), training=True)

            # 손실 계산
            loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)

            # 클래스 가중치 적용
            class_weight_tensor = tf.convert_to_tensor(list(class_weights.values()), dtype=tf.float32)
            weights = tf.reduce_sum(class_weight_tensor * y_batch, axis=1)
            weighted_loss = tf.reduce_mean(loss * weights)

        # 그래디언트 계산
        gradients = tape.gradient(weighted_loss, model.trainable_variables)
        # 옵티마이저 적용
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 에포크당 손실 출력
    epoch_loss = tf.reduce_mean(weighted_loss)

    # 훈련 정확도 계산
    y_pred = tf.argmax(predictions, axis=1)  # 모델 예측 클래스
    y_true = tf.argmax(y_batch, axis=1)  # 실제 클래스
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))  # 정확도 계산

    # 검증 손실 및 정확도 계산
    val_predictions = model.predict(np.expand_dims(X_test, axis=-1))
    val_loss = tf.keras.losses.categorical_crossentropy(y_test, val_predictions)
    val_loss = tf.reduce_mean(val_loss)  # 평균 검증 손실
    val_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(val_predictions, axis=1), tf.argmax(y_test, axis=1)), tf.float32))

    print(
        f'Epoch {epoch + 1} Loss: {epoch_loss.numpy()}, Accuracy: {accuracy.numpy()}, Val Loss: {val_loss.numpy()}, Val Accuracy: {val_accuracy.numpy()}')

# 평가 및 결과
y_pred = model.predict(np.expand_dims(X_test, axis=-1))
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# 혼동 행렬 및 분류 보고서 생성 및 출력
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test_classes, y_pred_classes, target_names=le.classes_)
print("Classification Report:")
print(class_report)

# 모델 저장
model.save('model_final.h5')
