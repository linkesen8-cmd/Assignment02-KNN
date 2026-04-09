import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 設定網頁標題與風格
st.set_page_config(page_title="客戶借貸風險評估", layout="wide")
st.title("One-Against-All SVM 客戶借貸風險預測")

# 1. 產生更「亂」的 150 筆資料
# 增加 cluster_std (從 3.5 改到 6.5)，並讓中心點更接近
centers = [[48, 22], [62, 15], [75, 25]] 
X, y = make_blobs(n_samples=150, centers=centers, cluster_std=6.5, random_state=42)

# 2. 調整 SVM 參數，使用較小的 C 值 (例如 0.1 或 0.5)
# 這會讓模型允許一些點「出軌」，這才叫現實
clf = svm.SVC(kernel='rbf', C=0.5, gamma=0.01, decision_function_shape='ovr')
clf.fit(X, y)

# 3. 側邊欄：使用者輸入資料
st.sidebar.header("--- 輸入客戶新資料 ---")
input_income = st.sidebar.number_input("輸入收入 (萬)", min_value=0.0, max_value=100.0, value=60.0)
input_debt = st.sidebar.number_input("輸入負債 (萬)", min_value=0.0, max_value=100.0, value=10.0)

X_user = np.array([[input_income, input_debt]])
y_pred = clf.predict(X_user)[0]

# 定義標籤名稱與顏色
labels = {0: "高風險", 1: "低風險", 2: "待審查"}
risk_result = labels[y_pred]

# 4. 顯示結果 (及格綠色，不及格紅色)
# 假設「低風險」與「待審查」算及格，「高風險」算不及格
if risk_result == "高風險":
    st.markdown(f"### 預測結果：<span style='color:red'>{risk_result}</span>", unsafe_allow_html=True)
else:
    st.markdown(f"### 預測結果：<span style='color:green'>{risk_result}</span>", unsafe_allow_html=True)

st.write(f"輸入座標：收入 {input_income} 萬，債務 {input_debt} 萬")

# 5. 繪圖
fig, ax = plt.subplots(figsize=(10, 6))

# 畫出決策區域底色
xx, yy = np.meshgrid(np.linspace(30, 95, 500), np.linspace(-5, 50, 500))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.1, colors=['red', 'blue', 'orange'])

# 繪製原始資料點
colors = ['red', 'blue', 'orange']
for i, label in labels.items():
    mask = (y == i)
    ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=label, edgecolors='k', alpha=0.7)

# 繪製使用者輸入的點 (星星)
ax.scatter(input_income, input_debt, c='yellow', marker='*', s=300, edgecolors='black', label=f'輸入: {risk_result}')
ax.annotate(f"({input_income}, {input_debt})", (input_income + 1, input_debt + 1), weight='bold')

ax.set_title("One-Against-All SVM Prediction")
ax.set_xlabel("Income (x1)")
ax.set_ylabel("Debt (x2)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

# 將圖表顯示在 Streamlit 網頁上
st.pyplot(fig)
