import os
import csv

# 設定txt檔案所在的資料夾路徑
folder_path = 'output_test/without/labels'

# 設定CSV檔案路徑及標題
csv_file = 'csvFile/distance_test/objectsOutput_without.csv'
csv_columns = ['item', 'class', 'x_center', 'y_center', 'width', 'height']

# 開啟CSV檔案，準備寫入資料
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

    # 逐一處理每個txt檔案
    item = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            class_dict = {'0': False, '56': False, '62': False, '63': False}
            with open(os.path.join(folder_path, filename), 'r') as txtfile:
                lines = txtfile.readlines()
                for line in lines:
                    data = line.split()
                    class_label = data[0]
                    class_dict[class_label] = True
                    # 將資料寫入CSV檔案
                    writer.writerow({'item': item,
                                     'class': class_label,
                                     'x_center': data[1],
                                     'y_center': data[2],
                                     'width': data[3],
                                     'height': data[4]})
                # 檢查是否每個class都至少有一個
                for key, value in class_dict.items():
                    if not value:
                        writer.writerow({'item': item,
                                         'class': key,
                                         'x_center': None,
                                         'y_center': None,
                                         'width': None,
                                         'height': None})
            item += 1