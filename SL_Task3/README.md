# 📊 Regularisation Comparison Lab

## 📌 Project Overview

This project explores the impact of regularisation techniques on regression models, particularly in the presence of multicollinearity and high-dimensional data.

We compare:

* Linear Regression (no regularisation)
* Ridge Regression (L2 regularisation)
* Lasso Regression (L1 regularisation)

The goal is to understand how model performance and feature coefficients change as the regularisation strength (alpha) increases.

---

## 📂 Dataset

We used the Boston Housing dataset, which contains information about housing features such as:

* Number of rooms (RM)
* Crime rate (CRIM)
* Tax rate (TAX)
* Percentage of lower-income population (LSTAT)

Target variable:

* **MEDV (Median House Price)**

---

## 🔍 Objectives

* Train Linear, Ridge, and Lasso regression models
* Analyze the effect of different alpha values
* Visualize coefficient shrinkage (coefficient path plots)
* Evaluate models using MAE, RMSE, and R²
* Use LassoCV to find optimal alpha
* Identify features eliminated by Lasso

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Checked for missing values
* Standardized features using StandardScaler
* Split data into training and testing sets (80/20)

---

### 2. Models Used

* Linear Regression (baseline)
* Ridge Regression (multiple alpha values)
* Lasso Regression (multiple alpha values)

---

### 3. Regularisation Analysis

We trained Ridge and Lasso models across a range of alpha values:
alpha = [0.01, 0.1, 1, 10, 100]

For each alpha:

* Model was trained
* Coefficients were recorded
* Performance metrics were calculated

---

### 4. Coefficient Path Plot

We plotted how coefficients change with increasing alpha:

* Ridge: coefficients shrink gradually
* Lasso: some coefficients become exactly zero

---

### 5. Model Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

### 6. LassoCV (Automatic Alpha Selection)

We used LassoCV to:

* Automatically select optimal alpha
* Identify which features were removed (coefficients = 0)

---

## 📊 Results Summary

| Model  | Alpha | MAE | RMSE | R²  |
| ------ | ----- | --- | ---- | --- |
| Linear | -     | ... | ...  | ... |
| Ridge  | 0.1   | ... | ...  | ... |
| Ridge  | 1.0   | ... | ...  | ... |
| Lasso  | 0.1   | ... | ...  | ... |
| Lasso  | 1.0   | ... | ...  | ... |

---

## 🧠 Key Findings

* Linear Regression performed well but is sensitive to multicollinearity
* Ridge Regression reduced coefficient magnitude and improved stability
* Lasso Regression performed feature selection by shrinking some coefficients to zero
* LassoCV helped identify the most important features automatically

---

## 🏆 Best Model

The best model was selected based on R² and RMSE.

* Ridge performed best in terms of stability and generalization
* Lasso provided a simpler model by eliminating less important features

---

## 📌 Features Eliminated by Lasso

Example:

* CHAS
* DIS
* INDUS

(Actual results depend on dataset run)

---

## 📈 Conclusion

Regularisation plays a critical role in improving model performance:

* Ridge is useful when all features are relevant
* Lasso is useful for feature selection
* Proper alpha tuning is essential for optimal results

---

## 🚀 Future Improvements

* Use ElasticNet (combination of Ridge + Lasso)
* Apply cross-validation for all models
* Test on larger, real-world datasets

---

## 🛠️ Tools & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## 👨‍💻 Author

Your Name
