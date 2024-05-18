import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('csvFile/distance_test/objectsOutput_without.csv')

# 仅保留class为0和56的数据
filtered_df = df[(df['class'] == 0) | (df['class'] == 56)]

# 初始化一个空的DataFrame来存储计算的距离
distances_df = pd.DataFrame(columns=['item', 'distance', 'normalized distance'])

# 循环计算每个item的距离
for item in filtered_df['item'].unique():
    class_0_coords = filtered_df[(filtered_df['item'] == item) & (filtered_df['class'] == 0)].iloc[0]
    class_56_coords = filtered_df[(filtered_df['item'] == item) & (filtered_df['class'] == 56)].iloc[0]
    
    x0, y0 = class_0_coords['x_center'], class_0_coords['y_center']
    x56, y56, w56, h56 = class_56_coords['x_center'], class_56_coords['y_center'], class_56_coords['width'], class_56_coords['height']
    
    # 使用欧氏距离公式计算距离
    # distance = abs(x0 - x56)
    # distance = np.sqrt((x0 - x56)**2 + (y0 - y56)**2)
    distance = np.sqrt(((x0 - x56)/x56)**2 + ((y0 - y56)/y56)**2)
    
    # 将结果添加到DataFrame中
    # distances_df = pd.concat([distances_df, pd.DataFrame({'item': [item], 'distance': [distance], 'normalized distance': [distance/w56]})], ignore_index=True)
    distances_df = pd.concat([distances_df, pd.DataFrame({'item': [item], 'distance': [distance], 'normalized distance': [distance/(w56*h56)]})], ignore_index=True)

# 将结果保存到CSV文件中
distances_df.to_csv('csvFile/distance_test/EuclideanNMAndChairArea_without.csv', index=False)