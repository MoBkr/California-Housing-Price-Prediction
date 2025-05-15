# 🏠 California Housing Price Prediction

This project aims to predict California housing prices based on real-world census data using end-to-end machine learning pipelines. It focuses on preparing data for modeling, selecting features, training models, and evaluating results using regression techniques.

---

## 🎯 Project Objective

The goal of this project is to:
- Analyze housing data from California districts.
- Engineer meaningful features that improve model performance.
- Build a machine learning pipeline to predict `median_house_value`.
- Evaluate the model's accuracy using suitable metrics.

This project simulates a real-world machine learning scenario that applies data cleaning, feature engineering, modeling, and evaluation in a modular and reusable way.

---

## 🧠 Skills & Technologies Used

| Category             | Tools / Techniques                                            |
|----------------------|---------------------------------------------------------------|
| **Languages**        | Python                                                        |
| **Data Handling**    | Pandas, NumPy                                                 |
| **Visualization**    | Matplotlib, Seaborn                                           |
| **Preprocessing**    | Scikit-learn Pipelines, OneHotEncoding, FeatureUnion          |
| **Model Evaluation** | MAE, R² Score                                                 |
| **Model Tuning**     | GridSearchCV, RandomizedSearchCV *(ready for integration)*    |
| **Notebook & Script**| Jupyter Notebook, Modularized Python Scripts (utils.py)       |

---

## 🤖 Machine Learning Models

- ✅ **Linear Regression** – baseline model
- ✅ **Polynomial Regression** – via `PolynomialFeatures`
- 🔜 *(Ready for extension)*:
  - Ridge / Lasso / ElasticNet
  - Random Forest Regressor
  - XGBoost Regressor

---

## 📊 Evaluation Metrics

Two primary metrics were used to assess model performance:

| Metric | Description |
|--------|-------------|
| **MAE (Mean Absolute Error)** | Measures the average magnitude of prediction errors. Simple and easy to interpret. |
| **R² Score (Coefficient of Determination)** | Indicates how well the model explains variance in the target variable. Ranges from 0 to 1. |

These metrics were calculated using both train-test split and cross-validation for reliability.

---


---

## 🚀 Features

- Data cleaning and feature engineering:
  - `rooms_per_household`
  - `bedrooms_per_rooms`
  - `population_per_household`
- Handling missing values and categorical encoding
- Scikit-learn pipelines for modular preprocessing
- Reusable `new_process()` function for transforming new input data

-



