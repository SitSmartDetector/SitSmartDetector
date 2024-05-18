from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import numpy as np
import tensorflow as tf
import cv2
from run_model import detect, draw_prediction_on_image
import asyncio
import time

app = FastAPI()

# Load MLP models
body_model = tf.keras.models.load_model('./Body/Body_weights.best.keras')
feet_model = tf.keras.models.load_model('./Feet/Feet_weights.best.keras')
head_model = tf.keras.models.load_model('./Head/Head_weights.best.keras')
neck_model = tf.keras.models.load_model('./Neck/Neck_weights.best.keras')
shoulder_model = tf.keras.models.load_model('./Shoulder/Shoulder_weights.best.keras')

async def predict_body(input_tensor_reshaped):
    return body_model.predict(input_tensor_reshaped)

async def predict_feet(input_tensor_reshaped):
    return feet_model.predict(input_tensor_reshaped)

async def predict_head(input_tensor_reshaped):
    return head_model.predict(input_tensor_reshaped)

async def predict_neck(input_tensor_reshaped):
    return neck_model.predict(input_tensor_reshaped)

async def predict_shoulder(input_tensor_reshaped):
    return shoulder_model.predict(input_tensor_reshaped)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # start_time = time.time()
    try:
        # 从上传的文件中读取内容
        contents = await file.read()
        # 将读取的内容解码为图像
        image = tf.io.decode_jpeg(contents)

        # MoveNet
        person = detect(image)
        pose_landmarks = []
        for keypoint in person.keypoints:
            pose_landmarks += [keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
        pose_landmarks = np.array(pose_landmarks)

        # 创建 TensorFlow tensor，准备用于模型预测
        input_tensor = tf.constant(pose_landmarks, dtype=tf.float32)
        input_tensor_reshaped = tf.reshape(input_tensor, shape=(1, -1))

        # 并行执行五个预测任务
        body_pred_task = asyncio.create_task(predict_body(input_tensor_reshaped))
        feet_pred_task = asyncio.create_task(predict_feet(input_tensor_reshaped))
        head_pred_task = asyncio.create_task(predict_head(input_tensor_reshaped))
        neck_pred_task = asyncio.create_task(predict_neck(input_tensor_reshaped))
        shoulder_pred_task = asyncio.create_task(predict_shoulder(input_tensor_reshaped))

        body_prediction, feet_prediction, head_prediction, neck_prediction, shoulder_prediction = await asyncio.gather(
            body_pred_task, feet_pred_task, head_pred_task, neck_pred_task, shoulder_pred_task)

        # 绘制关键点在图像上
        image_np = draw_prediction_on_image(image.numpy(), person, crop_region=None,
                                            close_figure=False, keep_input_size=True)

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 将带有关键点的图像转换为JPEG字节流
        _, encoded_image = cv2.imencode('.jpg', image_bgr)
        encoded_image_bytes = encoded_image.tobytes()

        # 处理每个模型的预测结果
        body_result = 'Body Backward' if np.argmax(body_prediction) == 0 else 'Body Forward' if np.argmax(body_prediction) == 1 else 'Body Neutral'
        feet_result = 'Feet Ankle-on-knee' if np.argmax(feet_prediction) == 0 else 'Feet Flat'
        head_result = 'Head Bowed' if np.argmax(head_prediction) == 0 else 'Head Neutral' if np.argmax(head_prediction) == 1 else 'Head Tilt Back'
        neck_result = 'Neck Forward' if np.argmax(neck_prediction) == 0 else 'Neck Neutral'
        shoulder_result = 'Shoulder Hunched' if np.argmax(shoulder_prediction) == 0 else 'Shoulder Neutral' if np.argmax(shoulder_prediction) == 1 else 'Shoulder Shrug'
        
        # end_time = time.time()  # 记录结束时间
        # execution_time = end_time - start_time  # 计算执行时间
        # print(execution_time)
        # 返回带有关键点的图像以及预测结果
        # return Response(content=encoded_image_bytes, media_type="image/jpeg", headers={
        #     "body": body_result,
        #     "feet": feet_result,
        #     "head": head_result,
        #     "neck": neck_result,
        #     "shoulder": shoulder_result
        # })
        return {
            "body": body_result,
            "feet": feet_result,
            "head": head_result,
            "neck": neck_result,
            "shoulder": shoulder_result
        }
    except Exception as e:
        # 发生错误时返回错误信息
        raise HTTPException(status_code=500, detail=str(e))