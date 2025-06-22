import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.cm import ScalarMappable

# ==================== 数据加载与预处理 ====================
# 读取Excel数据（请根据实际情况修改文件路径和标签列名）
file_path = "E:\论文2数据集\G1\\adult\\adult.xlsx"  # 注意反斜杠转义
df = pd.read_excel(file_path)

# 分离特征和标签（假设标签列名为'Label'）
X = df.drop('Label', axis=1)  # 特征矩阵（所有非标签列）
y = df['Label']               # 目标标签

# 标准化特征数据（PCA对尺度敏感，必须标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== PCA降维 ====================
# 初始化PCA并降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)  # 降维后的二维数据

# ==================== 可视化设置 ====================
# 创建画布
plt.figure(figsize=(6, 5))

# 生成颜色映射（根据标签值自动分配颜色）
label_values = np.unique(y)  # 获取所有唯一标签值
norm = plt.Normalize(y.min(), y.max())  # 标准化标签值到[0,1]区间
cmap = plt.cm.get_cmap('viridis', len(label_values))  # 使用viridis色板

# 绘制环形空心散点图
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],       # 二维坐标数据
    s=60,                           # 点的大小
    facecolors='none',              # 空心效果核心参数
    edgecolor=cmap(norm(y)),        # 边框颜色（根据标签映射）
    linewidth=0.8,                  # 边框粗细
    alpha=0.5                       # 透明度（避免重叠遮挡）
)

# 添加颜色条（显示边框颜色与标签的对应关系）
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # 避免警告
plt.colorbar(sm, label='Class Label')

# ==================== 图表美化 ====================
plt.title('PCA 2D visualization of the adult dataset distribution', fontsize=14)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
plt.grid(linestyle='--', alpha=0.4)  # 虚线网格
plt.tight_layout()  # 自动调整布局

# 显示图表
plt.show()
