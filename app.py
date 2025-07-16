import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv("customer.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)
    return df

data = load_data()
numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']  # <- ✅ Add this here


# --- Train model ---
def train_model(data):
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[numeric] = scaler.fit_transform(X_train[numeric])
    X_test[numeric] = scaler.transform(X_test[numeric])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, scaler, X.columns.tolist(), X_test, y_test

# --- App UI ---
st.title("📉 Customer Churn Prediction App")

data = load_data()
model, scaler, features, X_test, y_test = train_model(data)

st.sidebar.header("🧾 Input Customer Data")

# --- User Inputs ---
tenure = st.sidebar.number_input("Tenure (in months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
internet = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
payment = st.sidebar.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
paperless = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])

# --- Encode Inputs ---
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract_One year': 1 if contract == 'One year' else 0,
    'Contract_Two year': 1 if contract == 'Two year' else 0,
    'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
    'InternetService_No': 1 if internet == 'No' else 0,
    'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
    'PaperlessBilling_Yes': 1 if paperless == 'Yes' else 0,
}

for col in features:
    if col not in input_data:
        input_data[col] = 0

input_df = pd.DataFrame([input_data])[features]
input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
)

# --- Prediction ---
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.session_state.predicted = True  # ✅ Remember prediction was made
    st.session_state.prediction = prediction
    st.session_state.probability = probability

# --- Show results and evaluation after prediction ---
if st.session_state.get("predicted", False):
    prediction = st.session_state.prediction
    probability = st.session_state.probability

    st.subheader("Prediction Result:")
    st.write("✅ Customer is likely to **stay**" if prediction == 0 else "⚠️ Customer is likely to **churn**")
    st.write(f"📊 Churn Probability: **{probability:.2%}**")

    st.markdown("---")
    st.subheader("📊 Model Evaluation Dashboard")

    eval_option = st.selectbox(
        "Choose an evaluation metric to view:",
        ("Select one...", "Confusion Matrix", "ROC Curve", "Classification Report")
    )

    # Evaluation logic
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train_eval[numeric] = scaler.fit_transform(X_train_eval[numeric])
    X_test_eval[numeric] = scaler.transform(X_test_eval[numeric])
    y_pred = model.predict(X_test_eval)
    y_proba = model.predict_proba(X_test_eval)[:, 1]

    if eval_option == "Confusion Matrix":
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y_test_eval, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'], ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

    elif eval_option == "ROC Curve":
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test_eval, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic (ROC)")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    elif eval_option == "Classification Report":
        from sklearn.metrics import classification_report
        report = classification_report(y_test_eval, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.background_gradient(cmap="YlGn", subset=['precision', 'recall', 'f1-score']))



    