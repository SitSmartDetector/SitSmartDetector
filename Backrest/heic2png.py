from PIL import Image
import pillow_heif
import os
import glob

# 源資料夾和目標資料夾的路徑
src_dir = r"/Volumes/SUNNY/Movenet/Backrest/heic/train"
dest_dir = r"/Volumes/SUNNY/Movenet/Backrest/train"

# 遍歷源資料夾中的每個子資料夾
for folder_name in os.listdir(src_dir):
    # 建立對應的目標資料夾路徑
    folder_path = os.path.join(src_dir, folder_name)
    dest_folder_path = os.path.join(dest_dir, folder_name)

    # 確保目標資料夾存在
    if not os.path.exists(dest_folder_path):
        os.makedirs(dest_folder_path)

    # 遍歷資料夾中的每個HEIF文件
    for file_path in glob.glob(os.path.join(folder_path, "*.HEIC")):
        # 讀取HEIF文件
        heif_file = pillow_heif.read_heif(file_path)

        # 轉換為Pillow圖像
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )

        # 獲得PNG格式的文件名
        base_name = os.path.basename(file_path)
        png_file_name = os.path.splitext(base_name)[0] + ".png"

        # 保存圖像
        image.save(os.path.join(dest_folder_path, png_file_name), "PNG")
print('Finish!')