import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model('model_final.h5')

# TFLiteConverter 설정
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 변환 옵션 설정
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

# 모델 변환
tflite_model = converter.convert()

# 변환된 모델 저장
with open('model_final_gru.tflite', 'wb') as f:
    f.write(tflite_model)