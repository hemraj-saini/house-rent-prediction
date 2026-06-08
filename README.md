# 🏠 House Rent Prediction System

A Machine Learning-based web application that predicts house rent using property details such as BHK, size, location, furnishing status, and other features.

## 🚀 Live Demo

[🚀 Launch Application](https://house-rent-prediction-1st.streamlit.app/)

## 📊 Project Overview

This project predicts house rent prices using a Gradient Boosting Regressor model.

### Features Used

* BHK
* Size
* Area Type
* City
* Furnishing Status
* Tenant Preferred
* Bathroom
* Point of Contact
* Day
* Month
* Day of Week
* Current Floor
* Total Floor
* Area Locality Frequency

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib
* Matplotlib
  
## 🤖 Machine Learning Workflow

1. Exploratory Data Analysis (EDA)
2. Data Cleaning
3. Feature Engineering
4. Feature Encoding
5. Model Comparison
6. Hyperparameter Tuning
7. Final Model Selection
8. Model Deployment

## 📈 Model Performance

| Model                | CV Score |
| -------------------- | -------: |
| GradientBoosting     |   0.8286 |
| HistGradientBoosting |   0.8268 |
| RandomForest         |   0.8222 |

### Best Model

GradientBoostingRegressor

## 📂 Project Structure

```text
house-rent-prediction/
│
├── house_rent_predict.py
├── house_prediction_model.joblib
├── area_locality_freq.pkl
├── house.jpg
├── requirements.txt
└── README.md
```

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run house_rent_predict.py
```

## 👨‍💻 Author

Hemraj Saini
