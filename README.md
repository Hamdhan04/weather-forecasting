# 🌦️ Time-Series Weather Forecasting Dashboard using Stacked LSTM Networks

An end-to-end, academic deep learning project featuring a **Stacked Long Short-Term Memory (LSTM)** neural network architecture, real **Open-Meteo ERA5** meteorological pipeline (Coordinates: `12.8675° N, 80.0713° E`), and an ultra-classy **AETHERIS Glassmorphism Web Platform** (FastAPI + Vanilla CSS + Chart.js).

---

## 🌟 Key Highlights & UI Upgrade
- **Classy Bespoke Web Interface**: Built with **AETHERIS** — a custom luxury dark-mode web application featuring deep midnight obsidian (`#07090e`), ambient radial lighting mesh, frosted acrylic glass cards (`backdrop-filter: blur(28px)`), and curated neon accents (Luminescent Cyan, Electric Amber, Royal Violet, and Emerald Mint).
- **Sub-Millisecond Inference**: Powered by a high-performance **FastAPI asynchronous backend** with direct PyTorch C++ tensor evaluation in memory.
- **Dynamic 60fps Charting**: Smooth Bezier splines and live uncertainty intervals powered by **Chart.js**.
- **Interactive Weather Simulator**: Real-time tactile glass sliders that pass custom meteorological inputs directly through the live PyTorch Stacked LSTM to compute forward thermal trajectories on the fly.


## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Deep Learning Architecture](#-deep-learning-architecture)
3. [Meteorological Data Pipeline](#-meteorological-data-pipeline)
4. [Evaluation Metrics & Equations](#-evaluation-metrics--equations)
5. [Project Directory Layout](#-project-directory-layout)
6. [Quickstart & Installation](#-quickstart--installation)
7. [Dashboard Features](#-dashboard-features)
8. [Academic Mini-Project Viva Q&A Guide](#-academic-mini-project-viva-qa-guide)

---

## 🔬 Project Overview

Accurate weather forecasting is inherently complex due to chaotic atmospheric dynamics, seasonal periodicities, and intricate non-linear interactions across variables such as temperature, relative humidity, barometric air pressure, and wind speed.

Traditional numerical weather prediction (NWP) models require massive supercomputing power and computationally intensive differential equation solvers. This project implements a **Multi-Variate Stacked LSTM Recurrent Neural Network** capable of learning temporal sequences, long-term seasonal rhythms, and diurnal oscillations to project multi-step forward weather horizons (12h, 24h, 48h ahead) in milliseconds.

---

## 🧠 Deep Learning Architecture

The model processes continuous time-series windows of **48 hourly time steps** (2 days of temporal context) across **4 meteorological features**:

```
Input Tensor: (Batch Size, 48 Time Steps, 4 Features)
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  LSTM Layer 1 (64 Hidden Units, return_sequences=True) │
└────────────────────────────────────────────────────────┘
   │ Output: (Batch Size, 48, 64)
   ▼
┌────────────────────────────────────────────────────────┐
│  Dropout Regularization Layer (Rate = 0.20)            │
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  LSTM Layer 2 (32 Hidden Units, return_sequences=False)│
└────────────────────────────────────────────────────────┘
   │ Output: (Batch Size, 32) [Final Recurrent State h_T]
   ▼
┌────────────────────────────────────────────────────────┐
│  Dropout Regularization Layer (Rate = 0.20)            │
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  Dense Bottleneck Layer (16 Units, ReLU Activation)   │
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  Dense Readout Projection (48 Units)                  │
└────────────────────────────────────────────────────────┘
   │
   ▼
Output Tensor: (Batch Size, 48 Time Steps Ahead)
```

### LSTM Mathematical Cell Formulation
For each time step $t$, the LSTM cell equations are computed as:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(Forget Gate)}$$

$$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(Input Gate)}$$

$$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(Candidate Cell State)}$$

$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \quad \text{(Updated Cell State)}$$

$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(Output Gate)}$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \quad \text{(Hidden / Output State)}$$

---

## 📊 Meteorological Data Pipeline (Open-Meteo Integration)

The project is integrated directly with the **Open-Meteo Historical Archive API (ERA5 Reanalysis)** for coordinates:
- **Latitude**: `12.8675° N`
- **Longitude**: `80.0713° E` (Chengalpattu / Chennai Region, Tamil Nadu, India)
- **Time Span**: Hourly records spanning 2 full years (17,520 records)
- **Features Captured**:
  - `temperature`: 2m air temperature (°C)
  - `humidity`: 2m relative humidity (%)
  - `pressure`: Surface barometric pressure (hPa)
  - `wind_speed`: 10m wind speed (km/h)

| Feature Name | Physical Unit | Description | Regional Meteorological Dynamics |
| :--- | :---: | :--- | :--- |
| **`temperature`** | °C | Ambient 2m dry-bulb temperature | Tropical climate; peak temperatures in May–June (~38-42°C), mild winters (~20-24°C) |
| **`humidity`** | % | Relative humidity | Maritime coastal influence; elevated humidity (70-95%) especially during Northeast Monsoon |
| **`pressure`** | hPa | Surface atmospheric pressure | Low-pressure cyclonic depressions during Oct–Dec Northeast Monsoon; ~1010-1014 hPa winter highs |
| **`wind_speed`** | km/h | 10m wind velocity | Daily afternoon Bay of Bengal sea breeze; gust surges during convective storm events |

### Test Set Performance on Real Open-Meteo Data
- **RMSE (Root Mean Squared Error)**: `1.32 °C`
- **MAE (Mean Absolute Error)**: `0.97 °C`
- **$R^2$ Score**: `0.7931`
- **MAPE**: `3.66%`


### Data Leakage Prevention
Feature scaling is handled strictly through **`MinMaxScaler` fitted exclusively on the Chronological Training Partition (70%)** before transforming the Validation (15%) and Test (15%) partitions.

---

## 📐 Evaluation Metrics & Equations

The model is evaluated using four standard regression benchmarks:

1. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
3. **Coefficient of Determination ($R^2$ Score)**:
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
4. **Mean Absolute Percentage Error (MAPE)**:
   $$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

---

## 🗂️ Project Directory Layout

```
dl mini/
├── data/
│   ├── weather_generator.py      # Synthetic multi-parameter meteorological data generator
│   └── weather_data.csv          # Multi-year hourly dataset (17,520 records)
├── model/
│   ├── lstm_model.py             # PyTorch Stacked LSTM Neural Network Module
│   ├── train.py                  # End-to-end training, validation, & metric evaluation
│   ├── saved_model.pt            # Serialized model checkpoint & weights
│   ├── saved_model.h5            # Model weights artifact
│   ├── feature_scaler.pkl        # Fitted MinMaxScaler for input features
│   ├── target_scaler.pkl         # Fitted MinMaxScaler for target temperature
│   └── model_metadata.json       # Training loss history, evaluation metrics, and test sample
├── utils/
│   ├── preprocessing.py          # Sliding window sequences & train/val/test splits
│   └── metrics.py                # Regression metrics (RMSE, MAE, R², MAPE)
├── server.py                     # FastAPI High-Performance Backend & Web Application
├── requirements.txt              # Dependency specification
└── README.md                     # Comprehensive academic documentation
```

---

## 🚀 Quickstart & Installation

### Step 1: Install Dependencies
Open your terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

*(Recommended Python version: 3.10 – 3.12)*

### Step 2: Launch the Classy Web Dashboard (FastAPI + Modern Web UI)
Start the high-performance application server:

```bash
python server.py
```

Open your browser at: **`http://localhost:8000`**



---

## 🖥️ Dashboard Features

1. **Top KPI Row**: Real-time cards displaying *Current Predicted Temperature*, *24h Forecast Delta*, *Ambient Humidity Index*, and *Model Confidence Score*.
2. **Forecast Studio**: Interactive Plotly graph comparing ground-truth test data against the LSTM predictions with an expandable **±1.96σ (95% CI) Uncertainty Envelope**.
3. **Multi-Step Horizon Switcher**: Instantly toggle predictions between **12h, 24h, and 48h ahead**.
4. **Performance & Residuals Tab**: 
   - Training vs. Validation Loss curves across epochs.
   - Prediction residual distribution histogram with zero-bias Gaussian centering.
   - Interactive gauge meters for $R^2$, RMSE, MAE, and Peak Error.
5. **Weather Simulator ("What-If" Engine)**: Interactive sliders (Temperature, Humidity, Pressure, Wind Speed) and quick presets (*Summer Heatwave*, *Winter Cold Snap*, *Low-Pressure Storm*) to simulate real-time neural forecasting.
6. **LSTM Architecture Blueprint**: Interactive Mermaid flowchart and parameter specification.

---

## 🎓 Academic Mini-Project Viva Q&A Guide

#### Q1: Why choose LSTM over traditional RNNs or ARIMA for weather prediction?
> **Answer**: Standard Recurrent Neural Networks suffer from the **vanishing gradient problem**, where gradients decay exponentially over long time steps ($T \ge 10$), making it impossible to capture long-term cyclical dependencies like seasonal and diurnal shifts. LSTMs solve this through dedicated additive **cell state channels** regulated by input, forget, and output gates. Compared to linear statistical models like ARIMA, LSTMs naturally capture complex non-linear cross-correlations across multi-variate features (e.g., how low barometric pressure amplifies humidity and temperature swings).

#### Q2: What is the purpose of `return_sequences=True` in Layer 1?
> **Answer**: In a stacked architecture, the first recurrent layer must output the hidden state at *every* intermediate time step $t=1 \dots T$, producing a 3D tensor of shape `(Batch, Lookback, Units)` so the subsequent LSTM layer can process a sequence of features across time. The second LSTM layer has `return_sequences=False` to aggregate the full sequence into a single 2D vector `(Batch, Units)` at the final time step $T$ before feeding the dense projection head.

#### Q3: How was data leakage prevented during normalization?
> **Answer**: Scalers (`MinMaxScaler`) were fitted **exclusively on the training partition (first 70% of chronological data)**. The validation and test sets were transformed using the parameters $(\min, \max)$ derived solely from the training set, guaranteeing that future test information never leaked into model optimization.
