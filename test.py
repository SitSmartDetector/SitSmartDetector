import numpy as np
import tensorflow as tf

# 載入 TFLite 模型並進行初始化
interpreter = tf.lite.Interpreter(model_path="./Body/pose_classifier.tflite")
interpreter.allocate_tensors()

# 取得輸入和輸出張量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 準備輸入資料
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
print(f"輸入張量的形狀: {input_shape}")
# 將資料放入輸入張量
interpreter.set_tensor(input_details[0]['index'], input_data)

# 執行推論
interpreter.invoke()

# 取得輸出結果
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
