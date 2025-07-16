# 📊 Customer Churn Prediction Web App

This project is a **machine learning-powered Streamlit web app** that predicts whether a customer is likely to churn based on key service and demographic features.

It includes:
- 🧠 Logistic Regression model
- 📈 Visualizations: Confusion Matrix, ROC Curve, Classification Report
- 🖥 A user-friendly web interface using Streamlit

---

## 🚀 Live Demo

( https://divya1511-bhardwaj-customer-prediction-app-cm3xrd.streamlit.app/)
---


## 🧠 Model Details

- **Algorithm**: Logistic Regression (`scikit-learn`)
- **Features**: tenure, monthly charges, contract type, internet service, payment method, etc.
- **Target**: `Churn` (Yes/No)
- **Encoding**: One-hot encoding of categorical variables
- **Scaler**: StandardScaler for numeric features

---

## 🖥 How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/divya1511-bhardwaj/customer-prediction.git
cd customer-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

---

## 🔍 Features

- Manual input for customer attributes via sidebar
- Real-time churn prediction with probability
- Visual evaluation metrics:
  - Confusion Matrix
  - ROC Curve
  - Classification Report (Precision, Recall, F1-score)

---

## 📚 Dataset

Based on the Telco Customer Churn dataset:
[Kaggle - Telco Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## ✅ TODO

- [ ] Add more models (Random Forest, XGBoost)
- [ ] Allow batch prediction from file
- [ ] Add SHAP or feature importance plot

---

## 🙌 Credits

- Developed using [Streamlit](https://streamlit.io)


