import tf2onnx
import tensorflow as tf

# .h5 형식의 모델 불러오기
model = tf.keras.models.load_model('wandb_model34.h5')

# tf2onnx를 사용해 ONNX로 변환
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

# ONNX 모델 저장
with open("wandb_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())


# import onnx
#
# # 모델 로드
# model = onnx.load('wandb_model.onnx')
#
# # 모델 그래프에서 입력 이름 출력
# for input in model.graph.input:
#     print(input.name)
#
#
# # 모델 그래프에서 출력 이름 출력
# for output in model.graph.output:
#     print(output.name)
