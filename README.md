# Volatility Predictor

A lightweight tool for forecasting stock volatility using ultra-high-frequency data. The project uses financial order book data to compute key metrics, apply time series models, and generate interpretable predictions to help traders make informed decisions.

## Overview

- **Objective:** Predict future volatility for over 100 stocks by processing real trading data.
- **Key Challenges:**
  - Handling ultra-high-frequency data
  - Choosing the right modeling approach (realized volatility vs. direct stock price modeling)
  - Evaluating models using multiple performance measures
- **Outcome:** A prediction tool that not only forecasts volatility but also provides clear insights into underlying market dynamics.

## Requirments

- Python 3.x
- Required libraries:

## Week 05 : Problem Formulation

- **Business Objective Defined:** Clarified how our volatility prediction model supports key business decisions.
- **Use Case Established:** Outlined how the solution will be integrated and utilized.
- **Current Solutions Reviewed:** Identified existing workarounds and their limitations.
- **Similar Problems Analyzed:** Researched comparable cases to inform our approach.
- **Manual Process Outlined:** Documented a step-by-step manual method for baseline understanding.
- **Data Exploration:** Mapped data sources, noted key assumptions, and outlined data owner priorities.
- **Model Options Explored:** Evaluated potential machine learning models suitable for our needs.
- **Performance Metrics Set:** Selected key metrics (e.g., MAE, RMSE) to measure model accuracy.

## Week 06: Summary

### 1. Framer Website & Presentation Flow Chart

- Learn Framer via crash course videos and explore templates.
- Build a basic website to present project findings.
- Design a preliminary flow chart outlining the final presentation structure.

### 2. Noise Reduction Research

- Investigate methods to reduce noise in the mid-point from the Bid-Ask Spread affecting realised volatility.
- Compare approaches such as adjusted realised volatility, the realised kernel estimator, and alternatives like the Lee-Ready algorithm.
- Assess their applicability to our two-level LOB data.

### 3. Feature Engineering

- Use a 1,000-row subset of LOB data to develop new features (e.g., midpoint, bid-ask spread, LOB spread curve, realised volatility, etc.).
- Document the rationale and methodology for each feature.

## Week 07: Pipeline Implementation & Model Testing

### 1. Feature Engineering Pipeline

- Developed a robust scikit-learn pipeline to automate feature engineering.
- Generated 40 new features from raw data to enhance predictive signals.
- Visualized feature correlations with a heatmap to uncover additional insights.

### 2. Exploratory Data Analysis

- Plotted histograms for all 50 data columns to assess distributions.
- Employed grid-search plots to determine optimal window sizes for the realized volatility column.
- Conducted scatter plots and PCA analysis to explore feature dependencies on realized volatility.

### 3. Model Testing

- Evaluated several machine learning models on the enriched dataset:
  - Autoencoder-MLP
  - CNNs with Gramian Angular Fields
  - LSTM-GARCH
  - SHL and MHL Neural Networks
  - Ridge Regression
- Compared model performance using plots to check for accuracy and signs of overfitting.
