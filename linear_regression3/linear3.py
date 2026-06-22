import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="🏥 Medical Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #E0F2FE;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #38BDF8;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Medical Cost Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2069/2069748.png", width=80)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Mode", ["📊 Data Overview", "🎯 Make Prediction", "📈 Model Analysis", "📋 Sample Data"])

# Load and prepare data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('insurance.csv')
        
        # Preprocess data
        data_processed = data.copy()
        data_processed['sex'] = data_processed['sex'].map({'male': 1, 'female': 0})
        data_processed['smoker'] = data_processed['smoker'].map({'yes': 1, 'no': 0})
        region_dummies = pd.get_dummies(data_processed['region'], prefix='region', drop_first=True)
        data_processed = pd.concat([data_processed, region_dummies], axis=1)
        data_processed.drop('region', axis=1, inplace=True)
        
        return data, data_processed
    except:
        st.error("❌ Error loading insurance.csv file. Please make sure it's in the same directory.")
        return None, None

data, data_processed = load_data()

if data is not None:
    # Train the model
    @st.cache_resource
    def train_model():
        if data_processed is not None:
            X = data_processed.drop('charges', axis=1)
            y = data_processed['charges']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return model, X_train.columns, X_test, y_test, y_pred, rmse, mae, r2
        return None, None, None, None, None, None, None, None
    
    model, feature_names, X_test, y_test, y_pred, rmse, mae, r2 = train_model()
    
    # DATA OVERVIEW PAGE
    if app_mode == "📊 Data Overview":
        st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        with col2:
            st.metric("Total Features", len(data.columns))
        with col3:
            missing = data.isnull().sum().sum()
            st.metric("Missing Values", missing, delta="Good" if missing == 0 else "Needs Attention")
        
        # Dataset preview
        st.markdown('<h3 class="sub-header">📋 Data Preview</h3>', unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True)
        
        # Show column information
        with st.expander("📖 Column Information"):
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes.values,
                'Unique Values': [data[col].nunique() for col in data.columns],
                'Sample Values': [str(data[col].unique()[:3]) for col in data.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Statistical summary
        st.markdown('<h3 class="sub-header">📈 Statistical Summary</h3>', unsafe_allow_html=True)
        st.dataframe(data.describe(), use_container_width=True)
    
    # MAKE PREDICTION PAGE
    elif app_mode == "🎯 Make Prediction":
        st.markdown('<h2 class="sub-header">🎯 Predict Medical Cost</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Patient Information")
            
            age = st.slider("Age", 18, 100, 30)
            sex = st.radio("Sex", ["male", "female"])
            bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
            children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
            smoker = st.radio("Smoker", ["no", "yes"])
            region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
            
            if st.button("🔮 Predict Cost", key="predict"):
                # Prepare input
                input_data = pd.DataFrame({
                    'age': [age],
                    'sex': [1 if sex == 'male' else 0],
                    'bmi': [bmi],
                    'children': [children],
                    'smoker': [1 if smoker == 'yes' else 0],
                    'region_northwest': [1 if region == 'northwest' else 0],
                    'region_southeast': [1 if region == 'southeast' else 0],
                    'region_southwest': [1 if region == 'southwest' else 0]
                })
                
                # Ensure all columns exist
                for col in feature_names:
                    if col not in input_data.columns:
                        input_data[col] = 0
                
                input_data = input_data[feature_names]
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Store in session state
                st.session_state.prediction = prediction
                st.session_state.patient_data = {
                    'age': age, 'sex': sex, 'bmi': bmi,
                    'children': children, 'smoker': smoker, 'region': region
                }
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Prediction Results")
            
            if 'prediction' in st.session_state:
                pred_value = st.session_state.prediction
                patient = st.session_state.patient_data
                
                st.metric("Predicted Annual Medical Cost", f"${pred_value:,.2f}")
                
                # Show risk factors
                st.markdown("### 🩺 Risk Assessment")
                
                risk_factors = []
                if patient['smoker'] == 'yes':
                    risk_factors.append("🚭 Smoking significantly increases costs")
                if patient['bmi'] >= 30:
                    risk_factors.append("⚖️ High BMI may increase costs")
                if patient['age'] >= 60:
                    risk_factors.append("👴 Age factor considered")
                if patient['bmi'] < 18.5:
                    risk_factors.append("📉 Low BMI may indicate health issues")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.info(factor)
                else:
                    st.success("✅ Healthy profile with lower risk factors")
                
                # Show comparison with average
                avg_cost = data['charges'].mean()
                difference = pred_value - avg_cost
                st.metric("Compared to Average", f"${difference:+,.2f}", 
                         delta="Higher than average" if difference > 0 else "Lower than average")
                
                # Save prediction option
                if st.button("💾 Save This Prediction"):
                    st.success("✅ Prediction saved (placeholder - implement database for permanent storage)")
            else:
                st.info("👈 Enter patient details and click 'Predict Cost'")
                st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=200)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show some sample predictions
        st.markdown('<h3 class="sub-header">📊 Sample Predictions</h3>', unsafe_allow_html=True)
        
        sample_cases = [
            {"Age": 25, "Sex": "male", "BMI": 22, "Children": 0, "Smoker": "no", "Region": "southwest"},
            {"Age": 45, "Sex": "female", "BMI": 30, "Children": 2, "Smoker": "yes", "Region": "northeast"},
            {"Age": 60, "Sex": "male", "BMI": 28, "Children": 0, "Smoker": "no", "Region": "southeast"}
        ]
        
        cols = st.columns(3)
        for idx, case in enumerate(sample_cases):
            with cols[idx]:
                # Prepare sample input
                sample_input = pd.DataFrame({
                    'age': [case['Age']],
                    'sex': [1 if case['Sex'] == 'male' else 0],
                    'bmi': [case['BMI']],
                    'children': [case['Children']],
                    'smoker': [1 if case['Smoker'] == 'yes' else 0],
                    'region_northwest': [1 if case['Region'] == 'northwest' else 0],
                    'region_southeast': [1 if case['Region'] == 'southeast' else 0],
                    'region_southwest': [1 if case['Region'] == 'southwest' else 0]
                })
                
                for col in feature_names:
                    if col not in sample_input.columns:
                        sample_input[col] = 0
                sample_input = sample_input[feature_names]
                
                sample_pred = model.predict(sample_input)[0]
                
                st.markdown(f"""
                <div style='background-color: #F3F4F6; padding: 1rem; border-radius: 10px;'>
                    <h4>Sample Case {idx+1}</h4>
                    <p><b>Age:</b> {case['Age']}<br>
                    <b>Sex:</b> {case['Sex']}<br>
                    <b>BMI:</b> {case['BMI']}<br>
                    <b>Children:</b> {case['Children']}<br>
                    <b>Smoker:</b> {case['Smoker']}<br>
                    <b>Region:</b> {case['Region']}</p>
                    <hr>
                    <h3>${sample_pred:,.0f}</h3>
                </div>
                """, unsafe_allow_html=True)
    
    # MODEL ANALYSIS PAGE
    elif app_mode == "📈 Model Analysis":
        st.markdown('<h2 class="sub-header">📈 Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"${rmse:,.0f}")
        with col3:
            st.metric("MAE", f"${mae:,.0f}")
        with col4:
            st.metric("Test Samples", len(y_test))
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted plot
            fig = go.Figure()
            
            # Sort data for better visualization
            y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
            sorted_indices = np.argsort(y_test_array)
            y_test_sorted = y_test_array[sorted_indices]
            y_pred_sorted = y_pred[sorted_indices]
            
            fig.add_trace(go.Scatter(
                x=np.arange(len(y_test_sorted)),
                y=y_test_sorted,
                mode='lines+markers',
                name='Actual Cost',
                line=dict(color='#3B82F6', width=2),
                marker=dict(size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=np.arange(len(y_pred_sorted)),
                y=y_pred_sorted,
                mode='lines+markers',
                name='Predicted Cost',
                line=dict(color='#EF4444', width=2),
                marker=dict(size=6, symbol='square')
            ))
            
            fig.update_layout(
                title='Actual vs Predicted Costs',
                xaxis_title='Patient Index (sorted by actual cost)',
                yaxis_title='Medical Cost ($)',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error distribution
            errors = y_test_array - y_pred
            fig2 = px.histogram(x=errors, nbins=30, 
                               title='Prediction Error Distribution',
                               color_discrete_sequence=['#8B5CF6'])
            fig2.add_vline(x=0, line_dash="dash", line_color="red")
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Feature importance
        st.markdown('<h3 class="sub-header">📊 Feature Importance</h3>', unsafe_allow_html=True)
        
        if model is not None:
            coefficients = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', ascending=False)
            
            fig3 = px.bar(coefficients, x='Coefficient', y='Feature', orientation='h',
                         title='Impact of Features on Medical Cost',
                         color='Coefficient',
                         color_continuous_scale='RdBu')
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            st.dataframe(coefficients, use_container_width=True)
    
    # SAMPLE DATA PAGE
    elif app_mode == "📋 Sample Data":
        st.markdown('<h2 class="sub-header">📋 Explore Sample Data</h2>', unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_age, max_age = st.slider("Age Range", 18, 100, (18, 100))
        with col2:
            smoker_filter = st.multiselect("Smoker Status", options=['yes', 'no'], default=['yes', 'no'])
        with col3:
            region_filter = st.multiselect("Region", 
                                          options=['southwest', 'southeast', 'northwest', 'northeast'],
                                          default=['southwest', 'southeast', 'northwest', 'northeast'])
        
        # Apply filters
        filtered_data = data[
            (data['age'] >= min_age) & 
            (data['age'] <= max_age) &
            (data['smoker'].isin(smoker_filter)) &
            (data['region'].isin(region_filter))
        ]
        
        st.metric("Filtered Records", len(filtered_data))
        
        # Display filtered data
        st.dataframe(filtered_data, use_container_width=True)
        
        # Download option
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_insurance_data.csv",
            mime="text/csv"
        )
        
        # Summary statistics
        st.markdown('<h3 class="sub-header">📈 Filtered Data Statistics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Cost", f"${filtered_data['charges'].mean():,.0f}")
        with col2:
            st.metric("Median Cost", f"${filtered_data['charges'].median():,.0f}")
        with col3:
            st.metric("Min Cost", f"${filtered_data['charges'].min():,.0f}")
        with col4:
            st.metric("Max Cost", f"${filtered_data['charges'].max():,.0f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
    <p>🏥 Medical Cost Prediction Dashboard | Built with Streamlit & Scikit-learn</p>
    <p>⚠️ This tool is for educational purposes. Always consult healthcare professionals for medical decisions.</p>
</div>
""", unsafe_allow_html=True)