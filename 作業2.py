import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. 網頁基本設定
# ==========================================
st.set_page_config(page_title="SVM 成績分類器", layout="wide")
st.title("Assignment: SVM 成績及格與不及格分類")
st.markdown("基於 **每月讀書時間** 與 **作業練習題數**，使用 **SVM (支持向量機)** 預測是否及格。")

# ==========================================
# 2. 側邊欄：讓使用者自訂 C 值 (0.001 ~ 100)
# ==========================================
st.sidebar.header("SVM 參數設定")
# 因為 C 值跨度很大，Streamlit 提供 select_slider 可以完美呈現對數級別的滑桿
c_options = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
c_value = st.sidebar.select_slider("選擇 C 值大小 (懲罰係數)", options=c_options, value=1.0)

# ==========================================
# 3. 按照作業要求生成 200 筆模擬資料
# ==========================================
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 200
    
    # 產生特徵 1：每月讀書時間 (140~200 小時)
    study_hours = np.random.uniform(140, 200, n_samples)
    # 產生特徵 2：每月作業練習題數 (30~100 題)
    practice_qs = np.random.uniform(30, 100, n_samples)
    
    X = np.column_stack((study_hours, practice_qs))
    
    # 隱藏規則：加入一些隨機雜訊，讓資料邊界有重疊，這樣調整 C 值才會有感
    score = (study_hours - 140)/60 * 0.5 + (practice_qs - 30)/70 * 0.5 + np.random.normal(0, 0.2, n_samples)
    
    # 標籤 y：1 代表及格 (Green), 0 代表不及格 (Red)
    y = np.where(score > 0.5, 1, 0)
    
    return X, y

X, y = load_data()

# ==========================================
# 4. 建立 SVM 模型與預測
# ==========================================
# 這裡使用 rbf 核函數 (高斯核)，因為它對 C 值的變化最敏感、視覺化最明顯
# 如果老師規定要用直線切，可以把 kernel 改成 'linear'
svm_model = SVC(kernel='rbf', C=c_value)
svm_model.fit(X, y)

# 預測訓練資料 (為了計算指標)
y_pred = svm_model.predict(X)

# ==========================================
# 5. 計算 Confusion Matrix 與效能指標
# ==========================================
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

# 指標計算與防呆機制 (避免分母為 0)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = (tp + tn) / len(y)
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
error_rate = 1 - accuracy

# ==========================================
# 6. 在網頁上呈現效能指標
# ==========================================
st.subheader(f"📊 當 C = {c_value} 時的效能指標")
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
st.subheader("🗺️ SVM 決策邊界視覺化 (Decision Boundary)")

fig, ax = plt.subplots(figsize=(10, 5))

# 設定網格點 (Meshgrid)
x_min, x_max = 135, 205
y_min, y_max = 25, 105
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.5))

# 預測背景網格顏色
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 作業要求：及格綠色(1)，不及格紅色(0)
cmap_light = ListedColormap(['#fcded9', '#dcedc1'])
ax.contourf(xx, yy, Z, cmap=cmap_light)

# 畫出 200 筆原始資料點
colors = ['red' if label == 0 else 'green' for label in y]
ax.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', s=25, alpha=0.8)

# 加上圖例
ax.scatter([], [], c='green', edgecolors='k', s=25, label='Pass (及格)')
ax.scatter([], [], c='red', edgecolors='k', s=25, label='Fail (不及格)')
ax.legend(loc='upper left')

ax.set_title(f"SVM Student Performance Classification (C={c_value})")
ax.set_xlabel("Monthly Study Time (140~200 hours)")
ax.set_ylabel("Monthly Practice Questions (30~100 Qs)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

st.pyplot(fig)
