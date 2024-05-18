from tensorflow import keras
import numpy as np

# 身體部位列表
body_parts = ['Body', 'Head', 'Neck', 'Shoulder', 'Feet']

# 針對每個部位進行模型載入和權重保存
for part in body_parts:
    # 假設每個模型的命名規則是 {part}.keras，例如 Body.keras
    model_path = f'./{part}/{part}_weights.best.keras'
    npz_path = f'{part}_weights.npz'

    # 載入模型
    model = keras.models.load_model(model_path)

    # 獲取模型的所有權重
    weights = model.get_weights()

    # 創建一個字典來儲存權重，其中每個權重使用層的名稱作為鍵
    weights_dict = {f'layer_{i}': weight for i, weight in enumerate(weights)}
    print({key: weights_dict[key].shape for key in weights_dict})
    # 存儲權重到NPZ文件
    np.savez(npz_path, **weights_dict)

    # 輸出成功訊息
    print(f'Weights for {part} have been saved to {npz_path}')
