# Medical Cost Prediction System

Machine learning web application that predicts medical insurance costs using Linear Regression. Built with Python, Scikit-learn, and Streamlit.

---

## Features

- Real-time cost predictions based on patient demographics
- BMI category detection (Underweight, Normal, Overweight, Obese)
- Interactive dashboard with sliders and dropdowns
- Cost breakdown with pie charts and bar charts
- Smoking vs Obesity impact comparison
- Responsive design with custom CSS

---

## Tech Stack

- Python 3.14
- Scikit-learn (Linear Regression)
- Streamlit (Web Framework)
- Pandas (Data Processing)
- NumPy (Numerical Operations)
- Plotly (Visualizations)
- Joblib (Model Serialization)

---

## Dataset

- Source: Medical Cost Personal Dataset (Kaggle)
- Total Records: 1,338
- Features: age, sex, bmi, children, smoker, region
- Target Variable: charges (medical costs)

---

## Model Performance

- Algorithm: Linear Regression
- R-Squared Score: 0.78
- Training Samples: 1,070
- Testing Samples: 268
- Total Features: 8

---

## Feature Coefficients

- smoker: +$23,651 (Increases cost significantly)
- children: +$425 (Increases cost)
- bmi: +$337 (Increases cost)
- age: +$257 (Increases cost)
- sex: -$18.59 (Decreases cost)
- region_northwest: -$371 (Decreases cost)
- region_southeast: -$658 (Decreases cost)
- region_southwest: -$810 (Decreases cost)

---

## Quick Start

Clone the repository:
```bash
git clone https://github.com/amitanathirse-wq/medical-cost-prediction.git
cd medical-cost-prediction

