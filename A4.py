import time
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE

# 从Excel文件中导入数据
input_file = "E:\论文2数据集\G3\Satisfied\satisfied.xlsx"
data = pd.read_excel(input_file)

# 分离特征和标签
X = data.drop('Label', axis=1).values
y = data['Label'].values

start_time = time.time()

# 应用Borderline-SMOTE算法，使用库函数
smote = BorderlineSMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

end_time = time.time()
run_time = end_time - start_time
print(f"算法运行时间为: {run_time} 秒")