# 🌧️ Attention-Based LSTM Rainfall Prediction Model with TIF Weighting

# 📌 Overview
This repository hosts an advanced rainfall prediction model that integrates Global Climate Model (GCM) ensembles with IMD (India Meteorological Department) observed data using a Bidirectional LSTM with Attention mechanism and a novel Truth-Indeterminacy-Falsehood (TIF) weighting scheme.
It provides robust monthly rainfall forecasts across 4,641 grid points covering India.

🚀 Key Features

Hybrid Data Integration: Combines IMD observations with multi-GCM ensemble predictions.

TIF Uncertainty Quantification: Dynamically weights GCM grids based on:

✅ Truth (reliability)

❌ Falsehood (noise)

❓ Indeterminacy (uncertainty)

Attention-LSTM Architecture: Captures long-term dependencies and highlights critical time steps.

Seasonal Feature Engineering: Includes cyclical encoding, monsoon flags, and quarterly indicators.

Autoregressive Forecasting: Recursive multi-step prediction with smart fallback for missing data.

Comprehensive Evaluation: Uses both standard (MSE, R²) and hydrological (NSE, KGE) metrics.

# 🔧 Methodology (Step-by-Step)

 1️⃣ Data Preparation
Input:

rainfall_imd_monthly.csv → shape: [n_months, 4641]

GCM ensemble predictions (multiple files, same shape)

Preprocessing:

Train/Test Split:

Train → 2015–2021

Test → 2022–2023

Add seasonal features:

Cyclical month encoding:
sin(2πm/12), cos(2πm/12)

Monsoon flag: June–September

Quarterly indicators: Q1–Q4

Normalize using RobustScaler (per grid point)

 2️⃣ GCM Ensemble Processing
Align GCM grids with IMD spatial points

Compute ensemble mean:
mean_gcm_train, mean_gcm_test

 3️⃣ TIF Weighting
Truth (T): Accuracy (MSE + Correlation + NSE)

Falsehood (F): Pattern mismatch (temporal inconsistency)

Indeterminacy (I): Error spread (MAE + max deviation)

Weighting Formula:  Nonlinear weighting (best-performing)

weights = (truth_scores ** 3) / sum(truth_scores ** 3)
Apply weights to GCM grids → downweight noisy areas

 4️⃣ Sequence Preparation
 
Create rolling windows of 12-month sequences:

Inputs:

Past observed rainfall → [n, 12, 4641]

TIF-weighted GCM data → [n, 12, 4641]

Seasonal features → [n, 12, 4]

Target:

Next month’s rainfall → [n, 4641]

5️⃣ Model Architecture

Model Name: Attention_LSTM

<img width="544" height="572" alt="image" src="https://github.com/user-attachments/assets/93829903-7ba6-4513-ae30-3c0de2ec3fe2" />


Input Stream	Shape
Observed Data	(None, 12, 4641)
GCM Data	(None, 12, 4641)
Seasonal Features	(None, 12, 4)

Layers:

Bidirectional(LSTM(128)) → for both observed and GCM inputs

Attention() → weights time steps

Concatenate + Dense(4641) → grid-wise rainfall prediction

Training Setup:

Early stopping

Learning rate reduction



6️⃣ Autoregressive Forecasting
Recursive prediction approach:

Use last 12 months to predict the next

Append prediction to input window

Repeat for 24 months (2022–2023)

 7️⃣ Evaluation
Metric	LSTM Value

RMSE	33.3512

MAE	20.6905

R²	0.8788

NRMSE	0.1049

MAPE	45.9219

NSE	0.8788

Correlation	0.9376

KGE	0.9016

<img width="774" height="833" alt="image" src="https://github.com/user-attachments/assets/0226f512-ab5f-4670-9b75-76e1afd95958" />


Time-series comparison: predicted vs observed

Spatial heatmaps of TIF scores

 # 💡 Key Insights
✅ TIF Improves Ensemble Reliability
High-Truth regions (e.g., Central India) → 22% lower RMSE

High-Falsehood regions → downweighted by ~40%

📅 Seasonality Matters
Monsoon months benefited most → +15% NSE

🎯 Attention Highlights Critical Months
July received 30% higher attention weights than average

# 🌟 Results
The Attention-LSTM model achieved an 88% accuracy in predicting rainfall when validated against IMD data, showing strong capability in capturing spatiotemporal patterns in monsoon behavior.

# 🙏 Acknowledgments
IMD – for observed rainfall datasets

CMIP6 – for GCM ensemble data

TensorFlow/Keras – for deep learning framework

# 🌟 Star this repo if you find it useful!
