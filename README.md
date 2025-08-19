# 🤖 AI Stock Trend Analyzer

An AI-powered stock trend analyzer built with Python, TensorFlow, and Streamlit to predict future stock prices.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-stock-trend-analyzer-ajjgnx5j769tclwgvtlrkr.streamlit.app/)  

## ✨ Description

This project is an end-to-end AI application that predicts the future closing price of stocks using a Gated Recurrent Unit (GRU) deep learning model. The application features an interactive web interface built with Streamlit, allowing users to select a stock and receive a live prediction based on the latest market data. This project demonstrates a complete workflow from data collection and model training to deployment as a web service.

## 📸 Demo

![App Screenshot](https://raw.githubusercontent.com/AKrishnaK05/AI-Stock-Trend-Analyzer/main/screenshot.png) 
---

## 🚀 Features

* **Live Data Fetching:** Gathers the latest stock data using the `yfinance` API.
* **Time-Series Prediction:** Utilizes a GRU neural network, trained with TensorFlow/Keras, to forecast the next day's price.
* **Multiple Specialist Models:** Supports predictions for multiple stocks, with a dedicated, specialist model trained for each one.
* **Interactive Web Interface:** A user-friendly UI built with Streamlit that allows for easy stock selection and clear visualization of results.

---

## 🛠️ Tech Stack

* **Language:** Python 3.11
* **Modeling:** TensorFlow, Keras, Scikit-learn
* **Data Handling:** Pandas, NumPy
* **Web App:** Streamlit
* **Data Source:** yfinance

---

## ⚙️ Setup and Installation

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/AKrishnaK05/AI-Stock-Trend-Analyzer.git
    cd AI-Stock-Trend-Analyzer
    ```

2.  **Create a Conda Environment**
    It's recommended to create an isolated environment to manage dependencies.
    ```bash
    conda create --name stock_env python=3.11
    conda activate stock_env
    ```

3.  **Install Dependencies**
    Install all the required libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♀️ Usage

1.  **To Train the Models:**
    To train the models from scratch, run the Jupyter Notebooks in order:
    * `Data_Preparation.ipynb`
    * `Model_Training_GRU.ipynb`

2.  **To Run the Web App:**
    Launch the Streamlit application with the following command:
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

    ├── models/                  # Folder for saved .h5 model files
    ├── scalers/                 # Folder for saved .joblib scaler files
    ├── Data_extraction.ipynb    # Notebook to download the initial dataset
    ├── Data_Preparation.ipynb   # Notebook to prepare data and create scalers
    ├── Model_Training_GRU.ipynb # Notebook to train the GRU models
    ├── app.py                   # The main Streamlit application script
    ├── requirements.txt         # List of Python dependencies
    └── README.md                # This file

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
