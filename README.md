# 🤖 AI Stock Trend Analyzer

An end-to-end machine learning application for time-series stock-price forecasting using specialist GRU models, live market data, and an interactive Streamlit interface.

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-stock-trend-analyzer-ajjgnx5j769tclwgvtlrkr.streamlit.app/)

## Overview

The project demonstrates an end-to-end ML workflow:

```text
Historical Market Data
        ↓
Data Extraction
        ↓
Data Preparation & Scaling
        ↓
GRU Model Training
        ↓
Saved Model + Scaler Artifacts
        ↓
Live Data Ingestion
        ↓
Inference
        ↓
Interactive Visualization
```

The application uses a dedicated GRU model for each supported stock. At inference time, the application fetches recent market data through `yfinance`, prepares the latest 60 closing-price observations using the corresponding scaler, runs the trained model, and displays the predicted next-day closing price.

## Key Features

- **Live data ingestion:** Fetches recent market data using `yfinance`.
- **Time-series ML:** Uses TensorFlow/Keras GRU networks for next-day closing-price forecasting.
- **Specialist models:** Supports separate trained model artifacts for different stock tickers.
- **Reusable preprocessing:** Stores and reloads ticker-specific scaling artifacts with `joblib`.
- **Interactive inference:** Streamlit interface for selecting a stock and generating a prediction.
- **Visualization:** Displays recent market data and closing-price trends.
- **Container-ready:** Includes a Docker configuration for reproducible application deployment.
- **CI validation:** Includes a lightweight GitHub Actions workflow for Python syntax validation.

## Machine Learning Workflow

### 1. Data Extraction

`Data_extraction.ipynb` retrieves historical market data from `yfinance` for model development.

### 2. Data Preparation

`Data_Preparation.ipynb` prepares the time-series data and creates the scaling artifacts required by the models.

### 3. Model Training

`Model_Training_GRU.ipynb` trains GRU-based neural networks for the supported stocks and saves the resulting model artifacts.

### 4. Inference

`app.py`:

1. Detects available trained models.
2. Loads the selected ticker's GRU model and scaler.
3. Fetches recent market data.
4. Uses the latest 60 closing prices as the inference window.
5. Applies the saved scaler.
6. Runs the GRU model.
7. Inversely transforms the prediction.
8. Displays the predicted next-day closing price and recent price chart.

## Engineering Practices

The project has been structured to separate the major stages of the ML lifecycle rather than keeping the entire workflow inside a single notebook. Model and preprocessing artifacts are persisted separately from the application code, while the Streamlit layer handles inference and visualization.

The repository also includes Docker and CI configuration to make the application easier to reproduce and validate.

## Tech Stack

| Area | Technologies |
|---|---|
| Language | Python 3.11 |
| ML/DL | TensorFlow, Keras, GRU |
| Data Processing | Pandas, NumPy |
| Preprocessing | Scikit-learn, Joblib |
| Data Source | yfinance |
| Application | Streamlit |
| Containerization | Docker |
| CI | GitHub Actions |

## Project Structure

```text
AI-Stock-Trend-Analyzer/
├── models/                  # Saved GRU model artifacts
├── scalers/                 # Saved preprocessing/scaler artifacts
├── Data_extraction.ipynb    # Historical data extraction
├── Data_Preparation.ipynb   # Time-series preparation and scaling
├── Model_Training_GRU.ipynb # GRU model training
├── app.py                   # Streamlit inference application
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
├── .github/
│   └── workflows/
│       └── ci.yml           # CI syntax validation
├── Screenshot.png           # Application screenshot
└── README.md
```

## Local Setup

### Option 1: Python

```bash
git clone https://github.com/AKrishnaK05/AI-Stock-Trend-Analyzer.git
cd AI-Stock-Trend-Analyzer

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Docker

```bash
docker build -t ai-stock-trend-analyzer .
docker run --rm -p 8501:8501 ai-stock-trend-analyzer
```

Then open `http://localhost:8501`.

## Training

To reproduce the ML workflow, run the notebooks in this order:

1. `Data_extraction.ipynb`
2. `Data_Preparation.ipynb`
3. `Model_Training_GRU.ipynb`

The application expects trained model and scaler artifacts in the `models/` and `scalers/` directories.

## Important Note

This project is an educational machine-learning application and is **not financial advice**. Stock-price forecasting is inherently uncertain, and model predictions should not be interpreted as guaranteed future prices.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
