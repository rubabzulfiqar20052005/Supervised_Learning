# Regression from Scratch and Scikit-learn Comparison

## Project Overview

This project demonstrates the implementation of Linear Regression using two approaches:

1. Manual implementation using the Ordinary Least Squares (OLS) method with NumPy
2. Built-in implementation using sklearn.LinearRegression

The goal is to understand the mathematical foundation of regression and compare it with a library-based approach.

---

## Dataset

We used the Boston Housing dataset, which contains features related to housing conditions such as:

* Number of rooms (RM)
* Crime rate (CRIM)
* Tax rate (TAX)
* Lower status population percentage (LSTAT)

Target variable:

* MEDV (Median House Price)

---

## Objectives

* Implement Simple Linear Regression manually using OLS
* Implement the same model using sklearn
* Visualize regression line over scatter plot
* Extend to Multiple Linear Regression
* Compare coefficients from both methods

---

## Methodology

### 1. Simple Linear Regression (Manual - OLS)

We selected a single feature (e.g., RM) and applied the OLS formula:

θ = (XᵀX)⁻¹ Xᵀy

Steps:

* Add bias term to X
* Compute parameters using matrix operations
* Predict values using θ

---

### 2. Simple Linear Regression (sklearn)

Used sklearn.LinearRegression:

* Fit model on same feature
* Generate predictions
* Compare with manual results

---

### 3. Visualization

* Scatter plot of actual data
* Regression line overlay
* Helps visualize model fit

---

### 4. Multiple Linear Regression

* Used all features from dataset
* Applied OLS formula manually
* Trained sklearn model for comparison

---

### 5. Coefficient Comparison

* Compared intercept and coefficients
* Verified both approaches produce similar results

---

## Results

### Simple Linear Regression

* Manual and sklearn results were nearly identical
* Minor differences due to numerical precision

### Multiple Linear Regression

* Coefficients closely matched between both implementations
* Confirmed correctness of manual implementation

---

## Key Insights

* OLS provides exact mathematical solution for Linear Regression
* sklearn simplifies implementation and handles optimizations
* Manual implementation improves understanding of matrix operations
* Multiple regression captures relationships across multiple features

---

## Conclusion

This project bridges theory and practice by implementing regression both manually and using sklearn. It demonstrates that library-based models are built on well-established mathematical foundations.

---

## Future Improvements

* Add Regularisation (Ridge/Lasso)
* Use cross-validation
* Handle multicollinearity
* Extend to polynomial regression

---

## Tools and Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## Author

Rubab
