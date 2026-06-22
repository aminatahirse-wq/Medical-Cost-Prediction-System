# linear3.py - COMPLETE WORKING VERSION WITH GRAPHS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page setup
st.set_page_config(page_title="Medical Cost Predictor", layout="wide")
st.title("🏥 Medical Cost Prediction Dashboard")

try:
    # 1. LOAD DATA
    data = pd.read_csv('insurance.csv')
    
    # 2. PREPROCESS
    data_processed = data.copy()
    data_processed['sex'] = data_processed['sex'].map({'male': 1, 'female': 0})
    data_processed['smoker'] = data_processed['smoker'].map({'yes': 1, 'no': 0})
    region_dummies = pd.get_dummies(data_processed['region'], prefix='region', drop_first=True)
    data_processed = pd.concat([data_processed, region_dummies], axis=1)
    data_processed.drop('region', axis=1, inplace=True)
    
    # 3. TRAIN MODEL
    X = data_processed.drop('charges', axis=1)
    y = data_processed['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 4. CALCULATE METRICS
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Show metrics
    st.subheader("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"${rmse:,.0f}")
    with col3:
        st.metric("MAE", f"${mae:,.0f}")
    with col4:
        st.metric("Test Samples", len(y_test))
    
    # 5. CREATE GRAPHS (MATPLOTLIB - ALWAYS WORKS)
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    y_pred_array = np.array(y_pred)
    errors = y_test_array - y_pred_array
    
    # Graph 1: Scatter plot
    st.subheader("📊 Actual vs Predicted Costs")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(y_test_array, y_pred_array, alpha=0.6, color='blue', s=50)
    ax1.plot([y_test_array.min(), y_test_array.max()], 
             [y_test_array.min(), y_test_array.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Cost ($)')
    ax1.set_ylabel('Predicted Cost ($)')
    ax1.set_title('Actual vs Predicted Medical Costs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    
    # Graph 2: Error histogram
    st.subheader("📉 Prediction Error Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(errors, bins=30, color='purple', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Prediction Error ($)')
    ax2.set_ylabel('Number of Patients')
    ax2.set_title('Distribution of Prediction Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    # 6. PREDICTION FORM
    st.subheader("🎯 Predict Your Medical Cost")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30)
        bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
        children = st.selectbox("Children", [0, 1, 2, 3, 4, 5])
    
    with col2:
        sex = st.radio("Sex", ["male", "female"])
        smoker = st.radio("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    
    if st.button("🔮 Predict Cost"):
        input_data = pd.DataFrame({
            'age': [age], 'sex': [1 if sex == 'male' else 0],
            'bmi': [bmi], 'children': [children],
            'smoker': [1 if smoker == 'yes' else 0],
            'region_northwest': [1 if region == 'northwest' else 0],
            'region_southeast': [1 if region == 'southeast' else 0],
            'region_southwest': [1 if region == 'southwest' else 0]
        })
        
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[X.columns]
        
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Annual Medical Cost: **${prediction:,.2f}**")
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")