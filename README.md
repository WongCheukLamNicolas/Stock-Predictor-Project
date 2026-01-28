# Stock Predictor Project

**Project Overview**  
This project implements and compares multiple deep learning models for stock price prediction using various neural network architectures and optimization algorithms. The system predicts stock prices using historical data and evaluates performance across different optimizers and model architectures.

**Models Implemented**
1. LSTM Models
    * Adamax Optimizer (Adamax.py): LSTM with Adamax optimizer (3 LSTM layers, 100 units each)
    * SGD Optimizer (SGD_optimizer.py): LSTM with Stochastic Gradient Descent optimizer
    * FTRL Optimizer (FTRL_optimizer.py): LSTM with Follow-The-Regularized-Leader optimizer (simplified architecture)

2. GRU Model 
  * Gated Recurrent Unit implementation comparing:
  * Adamax optimizer
  * FTRL optimizer
  * SGD optimizer

3. Simple RNN Model 
  * Comprehensive RNN implementation with extensive parameter tuning
  * Tests multiple configurations across different stocks

**Key Features**  
* Data Processing
* Data Source: Yahoo Finance (yfinance API)

**Stocks Analyzed:**  
* AAPL (Apple Inc.)
* 9843.T (Nitori Holdings Co Ltd - Japan Home Furnishings)
* Multiple other stocks in the RNN implementation
* Time Period: 2020-01-01 to 2024-12-31
* Preprocessing: MinMax normalization, train-test split (80-20)

**Evaluation Metrics**  
* Standard Metrics: RMSE, MAE  
* Success Rate Metrics:
  * Direction Prediction Accuracy (Up/Down)
  * Predictions within 5% of the actual price
  * 3-Day Trend Following Accuracy
  * Trading Simulation Returns
* Overall Success Score: Weighted average of direction, percentage, and trend accuracy

**Advanced Analysis**  
* Trading simulation with profit/loss calculation
* Window-based best success rate analysis
* Multiple random seed runs for statistical significance
* Parameter grid search (lookback periods, hidden sizes, epochs)

**Model Architectures**  
1. LSTM Architecture (Adamax/SGD)
    * Input → LSTM(100) → Dropout(0.2) → LSTM(100) → Dropout(0.2) → LSTM(100) → Dropout(0.2) → Dense(1)

2. GRU Architecture
    * Input → GRU(50) → GRU(50) → Dense(1)

3. RNN Architecture
    * Input → SimpleRNN(hidden_size) → SimpleRNN(hidden_size) → Dense(1)

**Performance Comparison**  
The project compares the performance of different optimizers across metrics:

**Optimizers Tested:**  
* Adamax: Adaptive learning rate, momentum-based
* SGD: Stochastic Gradient Descent with momentum
* FTRL: Follow-The-Regularized-Leader (typically for sparse data)

**Key Findings:**  
* Each optimizer shows different strengths in RMSE vs. success rate
* Best performance varies by stock and time period
* Trading simulation results show practical profitability

**References**  
* Yahoo Finance API (yfinance)
* TensorFlow/Keras for deep learning implementation
* Scikit-learn for preprocessing and metrics
* Academic literature on stock prediction with RNN/LSTM/GRU
