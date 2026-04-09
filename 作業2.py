import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 設定網頁標題與風格
st.set_page_config(page_title="客戶借貸風險評估", layout="wide")
st.title("One-Against-All SVM 客戶借貸風險預測")

# 1. 產生 150 筆資料 (三個群落)
# 中心點大約設在：高風險(45, 25), 低風險(65, 12), 待審查(80, 25)
centers = [[45, 23], [65, 12], [78, 25]]
X, y = make_blobs(n_samples=150, centers=centers, cluster_std=3.5, random_state=42)

# 2. 訓練 One-Against-All SVM (scikit-learn 的 SVC 預設即支援 OVR/OAA)
# decision_function_shape='ovr' 即是一對多策略
clf = svm.SVC(kernel='linear', C=1.0, decision_function_shape='ovr')
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
