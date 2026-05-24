# MY PROJECTS
# 📊 ML Projects — Kislay Jha

[![GitHub](https://img.shields.io/badge/GitHub-k98j-181717?logo=github)](https://github.com/k98j)

A portfolio of end-to-end machine learning projects spanning supervised learning, optimisation theory, and model deployment.

---

## 🫀 Heart Stroke Risk Prediction with Deployment
**Tech:** Python · KNN · Decision Trees · SVM · Streamlit · Scikit-Learn

Predicts stroke risk from patient health metrics using three classifiers (KNN, Decision Tree, SVM), with systematic comparison via cross-validated grid search. Handles class imbalance through resampling. Evaluated with ROC-AUC, precision, recall, and F1-score to minimise false negatives in a clinical context. Deployed as an interactive Streamlit web app for real-time predictions.

**Key skills:** Classification · Hyperparameter tuning · Class imbalance · Deployment  
**[Live App →](http://127.0.0.1:8501/)**

---

## ⚙️ Gradient Descent from Scratch
**Tech:** Python · NumPy · Optimisation Theory

Implements batch, stochastic (SGD), and mini-batch gradient descent from first principles using only NumPy — no ML frameworks. Derives and codes analytical gradient computations for MSE loss, demonstrating mastery of the chain rule, learning rate sensitivity, and convergence dynamics. Includes cost-curve visualisations comparing convergence speed and stability across variants.

**Key skills:** Optimisation · Calculus · NumPy · Algorithm implementation from scratch

---

## 🚢 Titanic Survival Prediction
**Tech:** Python · Logistic Regression · EDA · Scikit-Learn

Binary classification on the Titanic dataset. Features systematic missing-value imputation and engineered variables (family size, title extraction). Logistic Regression with threshold tuning; evaluated with accuracy, F1, precision, recall, and ROC-AUC. Visualised survival patterns across passenger class, gender, and age cohorts.

**Key skills:** Binary classification · Feature engineering · EDA · Model evaluation

---

## 🚗 Ford Car Engine Size Prediction
**Tech:** Python · Linear Regression · Feature Engineering · EDA

Regression task predicting engine size from vehicle attributes. Conducted thorough EDA (distributions, correlations, outliers). Applied Label Encoding and One-Hot Encoding based on feature cardinality. Evaluated with RMSE and R².

**Key skills:** Regression · Encoding strategies · EDA · Feature selection


## 🧠 Perceptron from Scratch

**Tech:** Python · NumPy · Neural Networks Fundamentals

Built a binary classification perceptron entirely from scratch using NumPy, implementing forward propagation, weight updates, and activation logic without ML libraries. Demonstrates the mathematical foundations of neural networks, linear decision boundaries, and gradient-based learning through iterative training and visualisation of convergence behaviour.

**Key skills:** Python · Deep Learning · ANN · Backpropogation


## 📈 Time Series Analysis

**Tech:** Python · Pandas · Matplotlib · Statsmodels

Performed exploratory and statistical time series analysis on sequential data, including trend and seasonality decomposition, rolling statistics, stationarity testing, and forecasting techniques to uncover temporal patterns and generate predictive insights.

**Key skills:** Python · Time Series · Statistics


## 📈 Kernel — AI Financial Analyst Dashboard  
**Tech:** Python · Flask · DSPy · OpenAI · yfinance · pandas · ReportLab

Analyzes public stock tickers and portfolios using live market data, financial metrics, and AI-generated insights. Retrieves one-year price history, computes return, volatility, Sharpe proxy, drawdown, moving averages, correlation, valuation ratios, and portfolio concentration. Uses DSPy reasoning chains with OpenAI to generate structured analyst-style summaries, market reads, risk commentary, and educational recommendations. Includes portfolio weight analysis, risk-balanced allocation suggestions, momentum-tilted weights, conversational financial queries, and automated PDF report generation.

**Key skills:**  Financial analytics · Flask deployment · DSPy · LLM reasoning · Portfolio analysis · PDF reporting

**Live App → (http://127.0.0.1:5000/)**


## Descriptions for other interesting projects
- PCA Dimensionality Reduction — Applied Principal Component Analysis to reduce feature dimensionality while preserving maximum variance for improved visualisation and model efficiency.
- GridSearchCV — Demonstrated systematic hyperparameter optimisation using GridSearchCV to improve model performance through cross-validated tuning.
- Insurance Charge Prediction — Built a regression model to predict medical insurance charges using demographic and lifestyle attributes with detailed EDA and feature engineering.
- NLP with ML — Implemented a traditional machine learning NLP pipeline using text preprocessing, vectorisation, and classification techniques for textual data analysis.
- K-Means Clustering — Applied unsupervised K-Means clustering to identify hidden patterns and segment data points based on feature similarity.
