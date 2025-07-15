import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("customer.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)
    return df

# Train model
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
    return model, X.columns.tolist(), scaler

# App interface
st.title("📉 Customer Churn Prediction App")

data = load_data()
model, features, scaler = train_model(data)

st.sidebar.header("🧾 Input Customer Data")

# User input
tenure = st.sidebar.number_input("Tenure (in months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
internet = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
payment = st.sidebar.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
paperless = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])

# Manually encode inputs
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

# Add all missing features as 0
for col in features:
    if col not in input_data:
        input_data[col] = 0

# Create DataFrame and enforce correct column order
input_df = pd.DataFrame([input_data])[features]

# Scale numeric columns
input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction Result:")
    st.write("✅ Customer is likely to **stay**" if prediction == 0 else "⚠️ Customer is likely to **churn**")
    st.write(f"📊 Churn Probability: **{probability:.2%}**")


# charts and roc curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluate the model and show plots
def show_evaluation(X_test, y_test, model):
    st.subheader("📈 Model Evaluation")

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'], ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # --- Classification Report ---
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.background_gradient(cmap="RdYlGn", subset=['precision', 'recall', 'f1-score']))

# After training the model, save test data for evaluation
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train[numeric] = scaler.fit_transform(X_train[numeric])
X_test[numeric] = scaler.transform(X_test[numeric])

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save for both prediction and evaluation
features = X.columns.tolist()

# --- Show evaluation charts ---
show_evaluation(X_test, y_test, model)
