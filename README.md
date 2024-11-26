# Time Series Analysis and Forecasting

This project predicts gold prices based on historical data from 2011 to 2022, using both statistical and deep learning models. The goal is to forecast future gold prices based on past trends and patterns.

## Data Used
- **Source**: Investing.com
- **Time Period**: 2011-2022
- **Features**: Open, Close, High, Low, Price Changes, and Date

## Steps Involved

### 1. **Data Preprocessing**
   - Handling missing data and converting date columns into proper datetime format.
   - Feature extraction and scaling using `StandardScaler`.

### 2. **Data Visualization**
   - Visualizing price trends and seasonal patterns using various plotting libraries like `matplotlib`, `seaborn`, and `plotly`.

### 3. **Stationarity Test**
   - Applied the **KPSS Test** to check the stationarity of the time series.

### 4. **Modeling**
   - **SARIMA Model**: Time series forecasting using SARIMA (Seasonal ARIMA).
   - **LSTM Model**: Used Long Short-Term Memory (LSTM) neural networks to predict gold prices.

### 5. **Evaluation**
   - Evaluated the models using **mean squared error (MSE)** and visualized forecasts alongside actual values.

## Skills Demonstrated
- **Data Preprocessing**: Feature extraction and scaling.
- **Time Series Analysis**: Conducted stationarity tests and seasonal decomposition.
- **Modeling**: Developed and evaluated SARIMA and LSTM models for time series forecasting.
- **Visualization**: Used `matplotlib`, `seaborn`, and `plotly` for visual analysis.

## File Structure

```plaintext
Gold-Price-Prediction/
│
├── Python Code - Gold Price Prediction.py           # Main code for data preprocessing, modeling, and prediction
├── Presentation Slides - Gold Price Prediction.pdf  # Project presentation slides
│
└── README.md                                        # Project documentation
```
