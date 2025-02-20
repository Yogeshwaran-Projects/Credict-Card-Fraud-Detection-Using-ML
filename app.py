import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Set Streamlit page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide", page_icon="💳")

# Custom styles
st.markdown("""
    <style>
        .stButton > button {
            background-color: #004aad;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            padding: 12px 20px;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #002b7a;
        }
        .prediction-card {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
        .fraud {
            background-color: #ffcccc;
            color: #b30000;
            border: 2px solid #b30000;
        }
        .legit {
            background-color: #ccffcc;
            color: #006600;
            border: 2px solid #006600;
        }
        .typing {
            font-size: 20px;
            font-weight: bold;
            color: #004aad;
        }
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.2; }
            100% { opacity: 1; }
        }
        .dot {
            animation: blink 1.5s infinite;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #004aad;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar - Model Performance
st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"✅ **Training Accuracy:** {train_acc:.2%}")
st.sidebar.write(f"📌 **Testing Accuracy:** {test_acc:.2%}")

# Sidebar - Fraud Data Distribution
st.sidebar.header("📊 Fraud Data Distribution")
fig = px.bar(x=["Legitimate", "Fraud"], y=[len(legit_sample), len(fraud)], color=["Legitimate", "Fraud"],
             color_discrete_map={"Legitimate": "green", "Fraud": "red"},
             title="Fraud Data Distribution")
st.sidebar.plotly_chart(fig, use_container_width=True)

# User input section
st.markdown("### 📥 Enter Transaction Features")
st.info("Enter **comma-separated numbers** for the transaction features (e.g., `-1.2, 0.5, 3.1, -2.7, ...`).")

# Input field
input_values = st.text_input("🔢 Feature Values (comma-separated)", placeholder="Enter values here...")

# Centered Predict Button
col1, col2, col3 = st.columns([1, 2, 1])  # Button centered
with col2:
    predict_btn = st.button("🔍 Predict Transaction")

# Prediction logic
if predict_btn:
    try:
        # Convert input to DataFrame with correct column names
        features = pd.DataFrame([np.array(input_values.split(","), dtype=np.float64)], columns=X.columns)

        # Display AI analyzing animation
        with st.container():
            st.markdown("<h3 class='typing'>🤖 AI is analyzing<span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></h3>", unsafe_allow_html=True)

            # Loading bar animation
            progress_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.04)
                progress_bar.progress(percent + 1)

            time.sleep(0.5)  # Additional delay to feel real

        # Make prediction
        prediction = model.predict(features)

        # Display result
        if prediction[0] == 0:
            st.markdown("<div class='prediction-card legit'>✅ Legitimate Transaction</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='prediction-card fraud'>⚠️ Fraudulent Transaction Detected!</div>", unsafe_allow_html=True)
            st.warning("🚨 This transaction is flagged as **fraudulent**. Immediate investigation is recommended!")

            # AI alert animation
            st.markdown("""
                <div style="text-align: center;">
                    <lottie-player src="https://assets10.lottiefiles.com/packages/lf20_j9q9evul.json"  
                        background="transparent"  
                        speed="1"  
                        style="width: 250px; height: 250px;"  
                        loop  
                        autoplay>
                    </lottie-player>
                </div>
            """, unsafe_allow_html=True)

    except ValueError:
        st.warning("⚠️ Please enter valid numeric values separated by commas.")

# Transaction Summary (Expander)
with st.expander("📜 Transaction Summary"):
    st.markdown("""
        - **Legitimate Transactions** are safe and processed normally. ✅  
        - **Fraudulent Transactions** trigger an alert and may require manual review. 🚨  
        - **Machine Learning Model Accuracy:** Shows the model’s confidence in predictions. 📊  
    """)

# Expander: Show sample data
with st.expander("📂 View Sample Transactions"):
    st.dataframe(data.sample(5))

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; color: gray;'>
        🔒 Built with <strong>Streamlit</strong> | Machine Learning Powered
    </p>
""", unsafe_allow_html=True)
