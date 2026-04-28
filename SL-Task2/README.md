# Metrics Dashboard and Gradient Descent from Scratch

## Project Overview

This project focuses on understanding regression evaluation metrics and implementing Gradient Descent from scratch. The objective is to manually compute regression metrics using NumPy and compare them with Scikit-learn results, followed by building a regression model using Batch Gradient Descent.

---

## Objectives

* Train a Linear Regression model
* Manually compute evaluation metrics:

  * MAE (Mean Absolute Error)
  * MSE (Mean Squared Error)
  * RMSE (Root Mean Squared Error)
  * R² Score
  * Adjusted R²
* Validate results using sklearn metrics
* Implement Batch Gradient Descent from scratch
* Visualize cost reduction over iterations
* Demonstrate convergence of regression line

---

## Dataset

A synthetic dataset was generated using NumPy for controlled experimentation.

Example:

* X values generated using random distribution
* y values generated using a linear relationship with noise

---

## Methodology

### 1. Linear Regression Model

A basic Linear Regression model was trained using Scikit-learn to obtain predictions for evaluation.

---

### 2. Manual Metric Computation

The following metrics were computed manually using NumPy:

MAE:
Mean of absolute differences between actual and predicted values

MSE:
Mean of squared differences

RMSE:
Square root of MSE

R² Score:
Proportion of variance explained by the model

Adjusted R²:
Adjusted version of R² accounting for number of features

---

### 3. Verification with sklearn

All manually computed metrics were verified using sklearn.metrics:

* mean_absolute_error
* mean_squared_error
* r2_score

This ensured correctness of manual implementations.

---

### 4. Gradient Descent from Scratch

Batch Gradient Descent was implemented using the following steps:

* Initialize parameters (slope and intercept)
* Compute predictions
* Calculate error
* Update parameters using gradients
* Repeat for multiple iterations

---

### 5. Cost Function

The cost function used:
Mean Squared Error (MSE)

The cost was tracked at each iteration to observe convergence.

---

### 6. Visualization

Two main visualizations were created:

1. Cost Curve:

* Plot of cost vs iterations
* Shows how error decreases over time

2. Regression Line Animation:

* Demonstrates how the regression line improves over iterations
* Helps visualize convergence toward optimal solution

---

## Results

* Manual metric calculations matched sklearn results
* Gradient Descent successfully minimized cost
* Regression line converged to optimal fit
* Cost curve showed smooth decreasing trend

---

## Key Insights

* Understanding metrics manually provides deeper intuition
* Gradient Descent is the foundation of many ML algorithms
* Learning rate plays a crucial role in convergence
* Too high learning rate can cause divergence
* Too low learning rate slows training

---

## Conclusion

This project demonstrates both theoretical understanding and practical implementation of regression evaluation and optimization techniques. It highlights the importance of understanding core machine learning concepts beyond using libraries.

---

## Future Improvements

* Implement Stochastic Gradient Descent (SGD)
* Extend to Multiple Linear Regression
* Experiment with different learning rates
* Add regularization (Ridge/Lasso)

---

## Tools and Libraries

* Python
* NumPy
* Matplotlib
* Scikit-learn

---

## Author

Rubab
