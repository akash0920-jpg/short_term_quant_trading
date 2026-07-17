# Art of Compounding - Quantitative Trading Algorithm

## 1. Description

This project was developed for the "Art of Compounding" Finance Club competition. The primary objective is to build a profitable, automated quantitative trading algorithm utilizing 3 months of historical daily stock prices (Open, High, Low, Close) across 10 different stocks. 

**The Problem Statement & Constraints:**
* **Strict Maximum Holding Period:** Any open position must be squared off within a maximum of 3 days. 
* **The Goal:** Develop a Python program that ingests hidden daily market data, executes quantitative trading strategies without look-ahead bias, and outputs a daily position log detailing capital allocation and returns.

**The Strategic Approach:**
Initially, the algorithm attempted to predict the exact duration of an upcoming trend (a multi-class problem). However, due to the limited dataset and extreme market noise, this was fundamentally simplified. The final approach utilizes a Binary Classification model designed to answer one straightforward question: *Based on today's technical indicators, will tomorrow's closing price be strictly higher than today's?*. 

---

## 2. Model Exploration & Hyperparameter Tuning

Throughout the development phase, several models and data-balancing techniques were tested and subsequently discarded due to the noisy nature of financial time-series data:

* **Support Vector Machines (SVM):** SVM was initially tested, but the maximum accuracy achieved was around 40%. The dataset was too small and noisy for a margin-based classifier to find a clean hyperplane. Furthermore, the 10 stocks possessed distinct microstructures, making feature scaling—a strict requirement for SVM—highly inefficient.
* **Synthetic Minority Over-sampling Technique (SMOTE):** Attempted to balance class disparities by synthesizing data. SMOTE completely shattered the model's logic by generating synthetic noise that blurred clear decision boundaries, dropping accuracy significantly.
* **Deep Random Forest (max_depth=30):** Increasing the tree depth to 30 caused the model to severely overfit. The deep trees essentially memorized every single data point and minor market blip in the training set, causing the model to panic and default to a coin-flip accuracy on unseen data.

---

## 3. Final Model Selection & Algorithmic Working

### The Model
The final model selected is a highly constrained **Binary Random Forest Classifier**. Because Random Forest is a tree-based ensemble method, it seamlessly handles the different volatilities of the 10 stocks without requiring feature scaling.

**Hyperparameters Used:**
To prevent overfitting, the Random Forest was forced to act as a blunt instrument, requiring substantial evidence to generate a "Buy" signal.
* `n_estimators=300`
* `max_depth=5`
* `min_samples_split=20`
* `min_samples_leaf=10`
* `max_features='sqrt'`
* `class_weight='balanced'`
* `random_state=42`

### Feature Engineering
The model evaluates the market daily using the following rolling technical indicators:
* **SMA_3 & SMA_10:** Fast (3-day) and slow (10-day) Simple Moving Averages. 
* **MA_Diff:** The difference between SMA_3 and SMA_10 to capture immediate momentum shifts.
* **Volatility:** 5-Day rolling standard deviation to measure the erratic nature of recent price action.
* **RSI (14-Day):** Relative Strength Index to identify overbought or oversold market conditions.
* **Daily Return:** The percentage change from the previous day's close to today's close.

### The Execution Engine (State Machine)
The actual trading program operates as a daily State Machine with strict risk management rules:
* **Look-Ahead Bias Prevention:** Model predictions are processed at the Close of Day *T*. Generated Buy/Sell signals are queued and executed at the exact Open price of Day *T+1*.
* **Strict 3-Day Rule Enforcement:** The algorithm tracks the exact lifespan of every open position. If a stock is held for a 3rd consecutive day, the system physically overrides the machine learning model and forcefully squares off the trade at the Day 3 Closing Price.
* **Capital Protection:** The system utilizes a percentage-based allocation model, deploying exactly 10% of the total available portfolio per trade. This captures fractional gains and prevents the mathematical "death spiral" of depleting capital during drawdowns.

**Final Results:** This highly disciplined architecture achieved a generalized baseline accuracy of 53.12%, translating to an exploitable mathematical edge that generated a **+6.77% Total Portfolio Return** across 144 trades on unseen test data.
