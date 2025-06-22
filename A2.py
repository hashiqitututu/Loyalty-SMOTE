import time

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors


def safe_level_smote(X, y, N=100, k=5):
    minority_class = Counter(y).most_common()[-1][0]
    X_minority = X[y == minority_class]
    N = N // 100
    new_samples = []
    new_labels = []

    neighbors = NearestNeighbors(n_neighbors=k).fit(X)
    for i in range(len(X_minority)):
        nn_array = neighbors.kneighbors([X_minority[i]], return_distance=False)[0]
        safe_level = sum(y[nn_array] == minority_class) / k

        for _ in range(N):
            nn = np.random.choice(nn_array)
            diff = X[nn] - X_minority[i]
            gap = np.random.rand() * safe_level
            new_sample = X_minority[i] + gap * diff
            new_samples.append(new_sample)
            new_labels.append(minority_class)

    X_resampled = np.vstack((X, new_samples))
    y_resampled = np.hstack((y, new_labels))

    return X_resampled, y_resampled


# 从Excel文件中导入数据
input_file = "E:\论文2数据集\G3\Satisfied\satisfied.xlsx"
data = pd.read_excel(input_file)

# 分离特征和标签
X = data.drop('Label', axis=1).values
y = data['Label'].values

# 计算运行时间
start_time = time.time()

# 应用Safe-Level-SMOTE算法
X_resampled, y_resampled = safe_level_smote(X, y, N=200, k=5)

end_time = time.time()
run_time = end_time - start_time
print(f"算法运行时间为: {run_time} 秒")
