import time
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE  # 导入SMOTE函数

# 从Excel文件中导入数据
input_file = "E:\论文2数据集\G3\Satisfied\satisfied.xlsx"
data = pd.read_excel(input_file)

# 分离特征和标签
X = data.drop('Label', axis=1).values
y = data['Label'].values

start_time = time.time()
# 应用SMOTE算法，使用imblearn库中的SMOTE函数
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

end_time = time.time()
run_time = end_time - start_time

print("运行时间:", run_time, "秒")
