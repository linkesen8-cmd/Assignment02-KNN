import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. 網頁基本設定 (使用寬版排版)
# ==========================================
st.set_page_config(page_title="k-NN vs SVM 模型比較", layout="wide")
st.title("Assignment: k-NN vs SVM 決策邊界與效能比較")
st.markdown("左右對照 **k-NN (切比雪夫距離)** 與 **SVM (RBF 高斯核)** 在相同資料集下的分類表現。")

# ==========================================
# 2. 側邊欄：兩種模型的參數設定
# ==========================================
st.sidebar.header("⚙️ 參數設定")
st.sidebar.subheader("k-NN 參數")
k_val = st.sidebar.slider("選擇 k 值大小", min_value=1, max_value=20, value=3, step=1)

st.sidebar.divider()

st.sidebar.subheader("SVM 參數")
c_options = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
c_val = st.sidebar.select_slider("選擇 C 值大小 (懲罰係數)", options=c_options, value=1.0)

# ==========================================
# 3. 準備共用的資料與畫布
# ==========================================
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 200
    study_hours = np.random.uniform(140, 200, n_samples)
    practice_qs = np.random.uniform(30, 100, n_samples)
    X = np.column_stack((study_hours, practice_qs))
    score = (study_hours - 140)/60 * 0.5 + (practice_qs - 30)/70 * 0.5 + np.random.normal(0, 0.2, n_samples)
    y = np.where(score > 0.5, 1, 0)
    return X, y

X, y = load_data()

# 建立共用的網格點 (Meshgrid)
x_min, x_max = 135, 205
y_min, y_max = 25, 105
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))

cmap_light = ListedColormap(['#fcded9', '#dcedc1'])
colors = ['red' if label == 0 else 'green' for label in y]

# 定義一個計算並印出指標的工具函數，讓程式碼更乾淨
def display_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    err = 1 - acc
    
    st.write(f"**混淆矩陣:** TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("Precision", f"{prec:.4f}")
    m3.metric("Recall", f"{rec:.4f}")
    m4, m5, m6 = st.columns(3)
    m4.metric("Specificity", f"{spec:.4f}")
    m5.metric("F1-score", f"{f1:.4f}")
    m6.metric("Error Rate", f"{err:.4f}")

# ==========================================
# 4. 畫面一分為二：左右對照
# ==========================================
# ⭐️ 考點：使用 st.columns(2) 將畫面切成左右兩欄
col_knn, col_svm = st.columns(2)

# ----------------- 左半邊：k-NN -----------------
with col_knn:
    st.header(f"🔷 k-NN (k={k_val})")
    
    # 訓練 k-NN 模型
    knn = KNeighborsClassifier(n_neighbors=k_val, metric='chebyshev')
    knn.fit(X, y)
    y_pred_knn = knn.predict(X)
    
    # 顯示指標
    display_metrics(y, y_pred_knn)
    
    # 繪圖
    Z_knn = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig_knn, ax_knn = plt.subplots(figsize=(7, 5))
    ax_knn.contourf(xx, yy, Z_knn, cmap=cmap_light)
    ax_knn.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', s=25, alpha=0.8)
    ax_knn.set_title("k-NN Decision Boundary (Chebyshev)")
    ax_knn.set_xlabel("Study Time")
    ax_knn.set_ylabel("Practice Qs")
    st.pyplot(fig_knn)

# ----------------- 右半邊：SVM -----------------
with col_svm:
    st.header(f"🔶 SVM (C={c_val})")
    
    # 訓練 SVM 模型
    svm = SVC(kernel='rbf', C=c_val)
    svm.fit(X, y)
    y_pred_svm = svm.predict(X)
    
    # 顯示指標
    display_metrics(y, y_pred_svm)
    
    # 繪圖
    Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig_svm, ax_svm = plt.subplots(figsize=(7, 5))
    ax_svm.contourf(xx, yy, Z_svm, cmap=cmap_light)
    ax_svm.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', s=25, alpha=0.8)
    ax_svm.set_title("SVM Decision Boundary (RBF)")
    ax_svm.set_xlabel("Study Time")
    ax_svm.set_ylabel("Practice Qs")
    st.pyplot(fig_svm)
