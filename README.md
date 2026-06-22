Medical Cost Prediction System
A machine learning-powered web application that predicts individual medical insurance costs based on personal health metrics. Built with Python, Scikit-learn, and Streamlit.

Features
Core Features
Real-time Predictions: Enter patient data and get instant cost estimates

BMI Category Detection: Automatic health risk assessment (Underweight, Normal, Overweight, Obese)

Interactive Dashboard: User-friendly interface with sliders and dropdowns

Cost Breakdown: See exactly how each factor affects your total cost

Visualizations
Pie Chart: Shows percentage contribution of each factor

Bar Chart: Displays cost contributions in dollars

Distribution Chart: Compares your prediction to the dataset

Technical Features
Linear Regression model trained on 1,338 insurance records

Feature importance analysis

Smoking vs Obesity impact comparison

Responsive design with custom CSS

Tech Stack
Technology	Purpose
Python 3.14	Programming language
Scikit-learn	Machine Learning (Linear Regression)
Streamlit	Web application framework
Pandas	Data manipulation
NumPy	Numerical operations
Plotly	Interactive visualizations
Joblib	Model serialization
Project Structure
text
medical-cost-prediction/
├── medical_cost.py              # Streamlit application
├── medical_notebook.ipynb       # Jupyter notebook with full analysis
├── insurance.csv                # Dataset (1338 records)
├── medical_cost_model_full.pkl  # Trained model with coefficients
├── feature_names.pkl            # Feature names for model
├── medical_cost_analysis.png    # Data visualization
├── smoking_vs_obesity_analysis.png  # Bonus analysis
└── README.md                    # Project documentation
Dataset
Source: Medical Cost Personal Dataset (Kaggle)

Features:

age: Age of the individual

sex: Gender (male/female)

bmi: Body Mass Index

children: Number of children/dependents

smoker: Smoking status (yes/no)

region: Residential area (northeast, northwest, southeast, southwest)

charges: Individual medical costs billed by health insurance (target variable)

Dataset Size: 1,338 records

Model Performance
Metric	Value
Algorithm	Linear Regression
R-Squared Score	0.78
Features	8
Training Size	1,070 records
Testing Size	268 records
Feature Coefficients
Feature	Coefficient	Impact
smoker	+$23,651	Increases cost significantly
children	+$425	Increases cost
bmi	+$337	Increases cost
age	+$257	Increases cost
sex	-$18.59	Slightly decreases cost
region_northwest	-$371	Decreases cost
region_southeast	-$658	Decreases cost
region_southwest	-$810	Decreases cost
How to Run Locally
Prerequisites
Python 3.8 or higher

pip (Python package manager)

Installation
Clone the repository

bash
git clone https://github.com/amitanathirse-wq/medical-cost-prediction.git
cd medical-cost-prediction
Install dependencies

bash
pip install -r requirements.txt
Run the Streamlit app

bash
streamlit run medical_cost.py
Open your browser and navigate to

text
http://localhost:8501
Requirements
Create a requirements.txt file with:

text
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
plotly==5.15.0
joblib==1.3.2
Key Insights
Smoking increases medical costs by approximately 280%

Obesity increases medical costs by approximately 49%

Smoking is the single largest cost driver, adding approximately $23,651 on average

Author
Amina Tahir

GitHub: amitanathirse-wq

License
This project is for educational purposes only.

