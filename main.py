import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.optimizers import legacy  # legacy optimizers

# 1. Weights & Biases 초기화 및 프로젝트 설정
wandb.init(project="dog_behavior_prediction", entity="dksalstn1116-university-of-ulsan")

# 2. 데이터 불러오기
data = pd.read_csv('./dataset/DogMoveData.csv')

# 3. 제거할 행동 리스트
remove_behaviors = ['Bowing', 'Jumping', 'Tugging', 'Synchronization', 'Extra_Synchronization',
                    'Sniffing', 'Playing', 'Panting', 'Eating', 'Pacing', 'Drinking', 'Carrying object', '<undefined>',
                    'Trotting', 'Sitting', 'Lying chest', 'Shaking']

# 4. 데이터 필터링
filtered_movedata = data[~data['Behavior_1'].isin(remove_behaviors)]

# 5. Task별로 데이터 필터링
filter_conditions = (
        (filtered_movedata['Behavior_1'] == 'Walking') & (filtered_movedata['Task'] == 'Task walk') |
        (filtered_movedata['Behavior_1'] == 'Standing') & (filtered_movedata['Task'] == 'Task stand') |
        (filtered_movedata['Behavior_1'] == 'Galloping')
)

filtered_movedata = filtered_movedata[filter_conditions]

# 6. 행동별 샘플 개수 조정 및 데이터 증강
desired_count = 100000
sampled_data = []
le = LabelEncoder()
le.fit(filtered_movedata['Behavior_1'])

# 센서 열 이름을 리스트로 정의
sensor_columns = ['GNeck_x', 'GNeck_y', 'GNeck_z','ANeck_x', 'ANeck_y', 'ANeck_z']

for behavior in filtered_movedata['Behavior_1'].value_counts().index:
    class_data = filtered_movedata[filtered_movedata['Behavior_1'] == behavior]

    # 10만개를 넘는 경우 샘플링
    if len(class_data) > desired_count:
        sampled_data.append(class_data.sample(n=desired_count, random_state=42))

    else:
        insufficient_data = class_data.copy()

        # Galloping의 경우 5만 개까지만 증강
        if behavior == 'Galloping':
            while insufficient_data.shape[0] < 50000:
                noise = np.random.normal(0, 0.05, insufficient_data[sensor_columns].shape)
                augmented_data = insufficient_data.copy()
                augmented_data[sensor_columns] += noise
                insufficient_data = pd.concat([insufficient_data, augmented_data])

            # 5만개로 맞추고 추가하지 않음
            sampled_data.append(insufficient_data[:50000])

        else:
            sampled_data.append(insufficient_data)

final_data = pd.concat(sampled_data)

# print(final_data.Behavior_1.value_counts())


# 데이터 전처리
activity_column = 'Behavior_1'
sensor_data = final_data[sensor_columns].copy()
activity_data = final_data[activity_column].copy()
sensor_data.fillna(0, inplace=True)
activity_data_encoded = le.transform(activity_data)

scaler = MinMaxScaler(feature_range=(-1, 1))
sensor_data[sensor_columns] = scaler.fit_transform(sensor_data[sensor_columns])

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(sensor_data, activity_data_encoded, test_size=0.2, random_state=42,
                                                    stratify=activity_data_encoded)
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)


# 클래스 가중치 조정
class_weights = {
    0: 5.0,  # Galloping (뛰기) - 가장 높게 설정하여 예측 강화
    1: 2.0,  # Standing (정지) - 상대적으로 낮게 설정
    2: 3.0   # Walking (걷기) - 중간 정도의 가중치로 설정
}

# 모델 정의
model = Sequential()
model.add(Conv1D(64, 2, activation='selu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(GRU(64, activation='selu'))
model.add(Dense(32, activation='selu'))
model.add(Dense(3, activation='softmax'))


# 옵티마이저 설정
optimizer = legacy.Adam()  # M1/M2 Mac에서 최적화된 legacy optimizer 사용

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

    # WandB에 로그 기록
    wandb.log({
        "epoch": epoch + 1,
        "loss": epoch_loss.numpy(),
        "accuracy": accuracy.numpy(),  # 훈련 정확도 로그 기록
        "val_loss": val_loss.numpy(),  # 검증 손실 로그 기록
        "val_accuracy": val_accuracy.numpy()  # 검증 정확도 로그 기록
    })

# 평가 및 결과 WandB에 로그
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

# WandB에 혼동 행렬 및 분류 보고서 로그
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                    y_true=y_test_classes,
                                                    preds=y_pred_classes,
                                                    class_names=le.classes_),
    "classification_report": wandb.Table(dataframe=pd.DataFrame(
        classification_report(y_test_classes, y_pred_classes, target_names=le.classes_, output_dict=True)).transpose())
})

# 모델 저장
model.save('wandb_model35.h5')
wandb.finish()
