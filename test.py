import tensorflow as tf

# Load the models
body_model = tf.keras.models.load_model('./Body/Body_weights.best.keras')
feet_model = tf.keras.models.load_model('./Feet/Feet_weights.best.keras')
head_model = tf.keras.models.load_model('./Head/Head_weights.best.keras')
neck_model = tf.keras.models.load_model('./Neck/Neck_weights.best.keras')
shoulder_model = tf.keras.models.load_model('./Shoulder/Shoulder_weights.best.keras')

# Save the models in .tf and .h5 formats
body_model.save('./Body/Body_weights.best.tf', save_format='tf')
body_model.save('./Body/Body_weights.best.h5', save_format='h5')

feet_model.save('./Feet/Feet_weights.best.tf', save_format='tf')
feet_model.save('./Feet/Feet_weights.best.h5', save_format='h5')

head_model.save('./Head/Head_weights.best.tf', save_format='tf')
head_model.save('./Head/Head_weights.best.h5', save_format='h5')

neck_model.save('./Neck/Neck_weights.best.tf', save_format='tf')
neck_model.save('./Neck/Neck_weights.best.h5', save_format='h5')

shoulder_model.save('./Shoulder/Shoulder_weights.best.tf', save_format='tf')
shoulder_model.save('./Shoulder/Shoulder_weights.best.h5', save_format='h5')


# import numpy as np
# import tensorflow as tf

# # 載入 TFLite 模型並進行初始化
# interpreter = tf.lite.Interpreter(model_path="./Body/pose_classifier.tflite")
# interpreter.allocate_tensors()

# # 取得輸入和輸出張量
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # 準備輸入資料
# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# print(f"輸入張量的形狀: {input_shape}")
# # 將資料放入輸入張量
# interpreter.set_tensor(input_details[0]['index'], input_data)

# # 執行推論
# interpreter.invoke()

# # 取得輸出結果
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)
