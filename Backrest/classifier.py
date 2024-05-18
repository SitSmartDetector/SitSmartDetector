import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# 讀取訓練數據
# train_with_df = pd.read_csv('csvFile/distance_train/X_with.csv')
# train_without_df = pd.read_csv('csvFile/distance_train/X_without.csv')
# train_with_df = pd.read_csv('csvFile/distance_train/Euclidean_with.csv')
# train_without_df = pd.read_csv('csvFile/distance_train/Euclidean_without.csv')
train_with_df = pd.read_csv('binary_classification/csvFile/distance_train/EuclideanNM_with.csv')
train_without_df = pd.read_csv('binary_classification/csvFile/distance_train/EuclideanNM_without.csv')

# train_with_df = pd.read_csv('csvFile/distance_train/XAndChairArea_with.csv')
# train_without_df = pd.read_csv('csvFile/distance_train/XAndChairArea_without.csv')
# train_with_df = pd.read_csv('csvFile/distance_train/EuclideanAndChairArea_with.csv')
# train_without_df = pd.read_csv('csvFile/distance_train/EuclideanAndChairArea_without.csv')
# train_with_df = pd.read_csv('csvFile/distance_train/EuclideanNMAndChairArea_with.csv')
# train_without_df = pd.read_csv('csvFile/distance_train/EuclideanNMAndChairArea_without.csv')

# 添加標籤列
train_with_df['label'] = 'with'
train_without_df['label'] = 'without'

# 合併兩個訓練集
train_df = pd.concat([train_with_df, train_without_df], ignore_index=True)

# 分割特徵和標籤
X_train = train_df[['normalized distance']]
y_train = train_df['label']

# 初始化並訓練模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, 'binary_classification/logistic_regression_model.pkl')

# 讀取測試數據
# test_with_df = pd.read_csv('csvFile/distance_test/_with.csv')
# test_without_df = pd.read_csv('csvFile/distance_test/X_without.csv')
# test_with_df = pd.read_csv('csvFile/distance_test/Euclidean_with.csv')
# test_without_df = pd.read_csv('csvFile/distance_test/Euclidean_without.csv')
test_with_df = pd.read_csv('binary_classification/csvFile/distance_train/EuclideanNM_with.csv')
test_without_df = pd.read_csv('binary_classification/csvFile/distance_test/EuclideanNM_without.csv')

# test_with_df = pd.read_csv('csvFile/distance_test/XAndChairArea_with.csv')
# test_without_df = pd.read_csv('csvFile/distance_test/XAndChairArea_without.csv')
# test_with_df = pd.read_csv('csvFile/distance_test/EuclideanAndChairArea_with.csv')
# test_without_df = pd.read_csv('csvFile/distance_test/EuclideanAndChairArea_without.csv')
# test_with_df = pd.read_csv('csvFile/distance_test/EuclideanNMAndChairArea_with.csv')
# test_without_df = pd.read_csv('csvFile/distance_test/EuclideanNMAndChairArea_without.csv')

# 添加標籤列
test_with_df['label'] = 'with'
test_without_df['label'] = 'without'

# 合併兩個測試集
test_df = pd.concat([test_with_df, test_without_df], ignore_index=True)

# 分割特徵和標籤
X_test = test_df[['normalized distance']]
y_test = test_df['label']

# 預測
y_pred = model.predict(X_test)

# # 評估模型
# accuracy = accuracy_score(y_test, y_pred)
# # Compute precision
# precision = precision_score(y_test, y_pred, average='binary', pos_label='without')
# # Compute recall
# recall = recall_score(y_test, y_pred, average='binary', pos_label='without')
# # Compute F-measure
# f_measure = f1_score(y_test, y_pred, average='binary', pos_label='without')

# print("Precision:", precision)
# print("Recall:", recall)
# print("F-measure:", f_measure)
# print("Accuracy:", accuracy)


report = classification_report(y_test, y_pred, target_names=['with', 'without'])
print("Classification Report:\n", report)

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
labels = ['with', 'without']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()