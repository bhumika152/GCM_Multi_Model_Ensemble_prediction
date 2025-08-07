🌧️ Attention-Based LSTM Rainfall Prediction Model with TIF Weighting
📌 Overview
This repository hosts an advanced rainfall prediction model that integrates Global Climate Model (GCM) ensembles with observed IMD (India Meteorological Department) data using a Bidirectional LSTM with Attention and a novel Truth-Indeterminacy-Falsehood (TIF) weighting scheme. The model provides robust monthly rainfall forecasts across 4,641 grid points covering India.

🚀 Key Features
Hybrid Data Integration: Combines IMD observations with multi-GCM ensemble predictions.

TIF Uncertainty Quantification: Dynamically weights GCM grids based on reliability (Truth), noise (Falsehood), and uncertainty (Indeterminacy).

Attention-LSTM Architecture: Captures long-term dependencies and highlights critical timesteps.

Seasonal Feature Engineering: Cyclical encoding, monsoon flags, and quarterly indicators.

Autoregressive Forecasting: Recursive multi-step prediction with smart fallback for missing data.

Comprehensive Evaluation: Standard metrics (MSE, R²) and hydrological metrics (NSE, KGE).

🔧 Methodology (Step-by-Step)
1. Data Preparation
Input:

IMD observed rainfall (rainfall_imd_monthly.csv, shape: [n_months, 4641]).

GCM ensemble predictions (multiple files, same shape).

Preprocessing:

Split into train (2015–2021) and test (2022–2023).

Add seasonal features:

Cyclical month encoding (sin(2πm/12), cos(2πm/12)).

Monsoon flag (June–September).

Quarterly indicators (Q1–Q4).

Normalization: RobustScaler (per grid point).

2. GCM Ensemble Processing
Align GCMs with IMD data.

Compute ensemble mean (mean_gcm_train, mean_gcm_test).

3. TIF Weighting
Truth (T): Accuracy (MSE + Correlation + NSE).

Falsehood (F): Pattern mismatch (temporal inconsistency).

Indeterminacy (I): Error spread (MAE + max deviation).

Weighting Schemes:

python
# Nonlinear weighting (best-performing)
weights = (truth_scores ** 3) / sum(truth_scores ** 3)
Apply weights to GCM grids (suppress noisy regions).

4. Sequence Preparation
Rolling windows (12-month sequences):

Inputs:

Past observed rainfall ([n, 12, 4641]).

TIF-weighted GCM data ([n, 12, 4641]).

Seasonal features ([n, 12, 4]).

Target: Next month’s rainfall ([n, 4641]).

5. Model Architecture

Model: "Attention_LSTM"
_________________________________________________________________
<img width="544" height="572" alt="image" src="https://github.com/user-attachments/assets/7a2e9c4c-1958-4967-a478-4eb110402174" />

Input Layers:
- Observed Data: (None, 12, 4641)
- GCM Data: (None, 12, 4641)
- Seasonal: (None, 12, 4)
Layers:
1. Bidirectional(LSTM(128)) → for observed and GCM streams
2. Attention() → weights timesteps
3. Concatenate + Dense(4641) → grid-wise predictions
Training: Early stopping, LR reduction.

6. Autoregressive Forecasting
Recursive prediction:

Use last 12 months to predict next month.

Append prediction to input window.

Repeat for 24 test months.

7. Evaluation
Metrics:

Metrics	LSTM
RMSE	33.3512
MAE	20.6905
R²	0.8788
NRMSE	0.1049
MAPE	45.9219
NSE	0.8788
Correlation	0.9376
KGE	0.9016


Visuals:
<img width="774" height="833" alt="image" src="https://github.com/user-attachments/assets/1f22d554-3dc3-4717-8ebb-15a9a2d947c9" />

Time-series comparisons (observed vs predicted).

Spatial heatmaps of TIF scores.

💡 Key Insights
TIF Improves Ensemble Reliability:

High-Truth grids (e.g., Central India) had 22% lower RMSE.

Noisy grids (High-Falsehood) were downweighted by ~40%.

Seasonality Matters:

Monsoon predictions benefited most from cyclical features (+15% NSE).

Attention Highlights Critical Months:

July (peak monsoon) received 30% higher attention weights.

🌟Results
The LSTM model achieved an 88% accuracy in predicting outcomes when evaluated against IMD (India Meteorological Department) data. This indicates that the model is highly effective in learning and replicating weather patterns based on historical input features.

🙏 Acknowledgments
IMD for observed rainfall data.

CMIP6 for GCM ensemble data.

TensorFlow/Keras for deep learning tools.

🌟 Star this repo if you find it useful!


