import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. 網頁全螢幕設定 (符合老師截圖的寬比例)
# ==========================================
st.set_page_config(page_title="k-NN Moon Shape 分類器", layout="wide")

# ==========================================
# 2. 準備 Moon Shape 資料
# ==========================================
@st.cache_data
def load_data():
    # 生成資料 (n=200 代表各類別 100 筆)，調整 noise 讓點散佈得像截圖
    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
    return X, y

X, y = load_data()

# ==========================================
# 3. 版面佈局：為了讓「圖在上、拉桿在下」，我們先用佔位符預留圖表的位置
# ==========================================
st.title("Assignment 02: k-NN Moon Shape 資料分類")
plot_placeholder = st.empty()

# 在下方放置拉桿 (範圍 1~20)
st.write("") 
k = st.slider("選擇 k 值大小", min_value=1, max_value=20, value=3, step=1)

# ==========================================
# 4. 訓練 k-NN 模型 (Chebyshev 距離)
# ==========================================
knn = KNeighborsClassifier(n_neighbors=k, metric='chebyshev')
knn.fit(X, y)

# 拿訓練集自己預測自己 (符合老師 k=1 時準確率 100% 的前提)
y_pred = knn.predict(X)

# ==========================================
# 5. 計算效能指標
# ==========================================
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
accuracy = (tp + tn) / len(y)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
error_rate = 1 - accuracy

# ==========================================
# 6. 開始精細繪製圖表 (Matplotlib)
# ==========================================
# 設定圖表大小，寬度拉長以符合截圖比例
fig, ax = plt.subplots(figsize=(12, 5))

# 設定網格點 (Meshgrid) 畫背景
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# 間距設小一點 (0.02) 讓 Chebyshev 的階梯邊緣夠細緻
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 預測網格顏色
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 自訂背景色塊：淺紫藍色、淺橘黃色
cmap_light = ListedColormap(['#949cf0', '#eed9a3'])
ax.contourf(xx, yy, Z, cmap=cmap_light)

# 畫出真實的資料點
ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=20, label='Class 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='orange', s=20, label='Class 1')

# 設定軸標籤
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# 將 Legend (圖例) 移到圖表外部右側
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)

# 將效能指標的文字浮動標註在圖表內部右上角
metrics_text = (
    f"Accuracy={accuracy:.3f}\n"
    f"Precision={precision:.3f}\n"
    f"Recall={recall:.3f}\n"
    f"Specificity={specificity:.3f}\n"
    f"F1={f1_score:.3f}\n"
    f"Error={error_rate:.3f}"
)
ax.text(0.98, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', color='#555555')

# 隱藏上方和右方的框線
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 將畫好的圖表丟回剛剛設定的上方「佔位區」
plot_placeholder.pyplot(fig)
