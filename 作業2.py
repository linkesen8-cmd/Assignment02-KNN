import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. 網頁基本設定
# ==========================================
st.set_page_config(page_title="學生成績 k-NN 分類器", layout="wide")
st.title("Assignment 03: 成績及格與不及格分類 (k-NN)")
st.markdown("基於 **每月讀書時間** 與 **作業練習題數**，使用 **Chebyshev 距離 ($p=\infty$)** 預測是否及格。")

# ==========================================
# 2. 側邊欄：讓使用者自訂 k 值 (1~20)
# ==========================================
st.sidebar.header("參數設定")
k = st.sidebar.slider("選擇 k 值大小", min_value=1, max_value=20, value=3, step=1)

# ==========================================
# 3. 按照作業要求生成 500 筆模擬資料
# ==========================================
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 500
    
    # 產生特徵 1：每月讀書時間 (140~200 小時)
    study_hours = np.random.uniform(140, 200, n_samples)
    # 產生特徵 2：每月作業練習題數 (30~100 題)
    practice_qs = np.random.uniform(30, 100, n_samples)
    
    # 將兩個特徵合併成 X 矩陣
    X = np.column_stack((study_hours, practice_qs))
    
    # 定義一個「隱藏規則」來決定是否及格 (加入一些隨機雜訊讓資料有真實感)
    # 假設讀書時間與練習題數加權後大於某個門檻就會及格
    score = (study_hours - 140)/60 * 0.5 + (practice_qs - 30)/70 * 0.5 + np.random.normal(0, 0.15, n_samples)
    
    # 標籤 y：1 代表及格 (Green), 0 代表不及格 (Red)
    y = np.where(score > 0.5, 1, 0)
    
    return X, y

X, y = load_data()

# ==========================================
# 4. 建立 k-NN 模型與預測 (Chebyshev)
# ==========================================
knn = KNeighborsClassifier(n_neighbors=k, metric='chebyshev')
knn.fit(X, y)

# 預測訓練資料 (為了計算指標，並觀察 k=1 時的完美擬合現象)
y_pred = knn.predict(X)

# ==========================================
# 5. 計算 Confusion Matrix 與效能指標
# ==========================================
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

# 避免分母為 0 的保護機制
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = (tp + tn) / len(y)
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
error_rate = 1 - accuracy

# ==========================================
# 6. 在網頁上呈現效能指標
# ==========================================
st.subheader(f"📊 當 k = {k} 時的效能指標")
st.write(f"**混淆矩陣 (Confusion Matrix):** TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Precision", f"{precision:.4f}")
col2.metric("Recall", f"{recall:.4f}")
col3.metric("Specificity", f"{specificity:.4f}")
col4.metric("Accuracy", f"{accuracy:.4f}")
col5.metric("F1-score", f"{f1_score:.4f}")
col6.metric("Error rate", f"{error_rate:.4f}")

st.divider()

# ==========================================
# 7. 繪製 2D 決策邊界圖
# ==========================================
st.subheader("🗺️ 決策邊界視覺化 (Decision Boundary)")

# 為了讓排版好看，我們跟前一個作業一樣開一個稍微寬一點的畫布
fig, ax = plt.subplots(figsize=(10, 5))

# 設定網格點 (Meshgrid)
x_min, x_max = 135, 205 # 稍微比 140~200 寬一點
y_min, y_max = 25, 105  # 稍微比 30~100 寬一點

# 注意：因為 X 軸的跨度較大，這裡的 step 設為 0.5 即可維持平滑
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.5))

# 預測背景網格顏色
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 作業要求：及格綠色(1)，不及格紅色(0)
# 背景色塊：0=淺紅 (#fcded9), 1=淺綠 (#dcedc1)
cmap_light = ListedColormap(['#fcded9', '#dcedc1'])
ax.contourf(xx, yy, Z, cmap=cmap_light)

# 畫出 500 筆原始資料點 (不及格=紅, 及格=綠)
colors = ['red' if label == 0 else 'green' for label in y]
ax.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', s=25, alpha=0.8)

# 加上圖例
ax.scatter([], [], c='green', edgecolors='k', s=25, label='Pass (及格)')
ax.scatter([], [], c='red', edgecolors='k', s=25, label='Fail (不及格)')
ax.legend(loc='upper left')

# 設定軸標籤
ax.set_title(f"Student Performance Classification (k={k}, Distance=Chebyshev)")
ax.set_xlabel("Monthly Study Time (140~200 hours)")
ax.set_ylabel("Monthly Practice Questions (30~100 Qs)")

# 隱藏右側和上方的框線讓圖表更美觀
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 顯示在 Streamlit 網頁上
st.pyplot(fig)
