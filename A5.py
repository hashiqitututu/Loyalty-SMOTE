import time
import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek

# 从 Excel 文件中导入数据
input_file = "E:\论文2数据集\G3\Satisfied\satisfied.xlsx"
data = pd.read_excel(input_file)

# 分离特征和标签
X = data.drop('Label', axis=1).values
y = data['Label'].values

start_time = time.time()

# 应用 SMOTETomek 算法
smote_tomek = SMOTETomek()
X_final, y_final = smote_tomek.fit_resample(X, y)

end_time = time.time()
run_time = end_time - start_time

print("运行时间:", run_time, "秒")
