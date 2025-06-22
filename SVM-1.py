import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, recall_score, accuracy_score, matthews_corrcoef, balanced_accuracy_score

# 1. 从Excel读取数据
file_path = 'E:\论文2数据集\G1\heart+disease\\smote.xlsx'# 替换为您的Excel文件路径
df = pd.read_excel(file_path)


# 2. 数据预处理（示例：假设数据包含特征列和目标列）
X = df.drop('Label', axis=1)
y = df['Label']

# 3. 数据标准化（如果需要）
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. 建立SVM模型
svm_model = SVC(kernel='rbf', probability=True)  # 根据需要选择核函数和其他参数
svm_model.fit(X_train, y_train)

# 6. 模型预测
y_pred = svm_model.predict(X_test)

# 7. 计算评估指标：F1值、AUC值、Recall、G-mean、MCC、ACC
f1 = f1_score(y_test, y_pred)
y_prob = svm_model.predict_proba(X_test)[:, 1]  # 获取正类的概率用于计算AUC
auc = roc_auc_score(y_test, y_prob)
recall = recall_score(y_test, y_pred)
g_mean = balanced_accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

# 8. 输出结果
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")
print(f"Recall: {recall}")
print(f"G-mean: {g_mean}")
print(f"MCC: {mcc}")
print(f"ACC: {acc}")