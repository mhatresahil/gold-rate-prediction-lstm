# gold-rate-prediction-lstm

# Gold Rate Prediction using Time Series Analysis and Neural Networks

## Overview
This project involves predicting the daily gold rate using time-series data and machine learning models, specifically focusing on Feedforward Neural Networks (FNN), Long Short-Term Memory networks (LSTM), and Gated Recurrent Units (GRU). The objective is to develop models capable of understanding temporal patterns and delivering robust predictions.

The dataset used contains gold rates indexed by date. Additionally, lag features (t-1 to t-5) were created to provide historical context for the models. The models were trained and evaluated on train and test sets using Mean Squared Error (MSE) as the loss metric.

---

## Problem
The problem involves predicting the daily gold rate based on historical data. Gold rates tend to show temporal dependencies, making time-series forecasting models like LSTM and GRU well-suited for the task.

---

## Data
The dataset consists of the following:
- **Date**: The date of each record.
- **Gold Rate**: The gold price corresponding to the date.
- **Lag Features**: Lagged values (t-1 to t-5) of the gold rate, generated to capture historical dependencies for the time-series models.

---

## Models

### 1. Feedforward Neural Network (FNN)
- **Architecture**: 
  - Input: 5 lag features
  - Layers: 5 → 10 → 20 → 20 → 10 → 5 → 1
  - Activation: Tanh
- **Loss**: MSE
- **Optimizer**: Adam (Learning Rate: 0.001)

### 2. Long Short-Term Memory (LSTM)
- **Architecture**:
  - Input: 5 lag features
  - Layers: LSTM (5 units) → Dense (1 output)
- **Loss**: MSE
- **Optimizer**: Adam (Learning Rate: 0.001)

### 3. Gated Recurrent Unit (GRU)
- **Architecture**:
  - Input: 5 lag features
  - Layers: GRU (10 units) → Dense (1 output)
- **Loss**: MSE
- **Optimizer**: Adam (Learning Rate: 0.001)

---

## Results
The table below summarizes the performance of each model after 100 epochs:

| Model                     | Train Loss | Test Loss |
|---------------------------|------------|-----------|
| Feedforward Neural Network | 0.0005     | 0.0006    |
| LSTM Network              | 0.0002     | 0.0006    |
| GRU Network               | 0.0002     | 0.0007    |

### Observations:
- The **LSTM model** delivers the lowest loss values and performs best in this task.
- The **GRU model** performs second best, slightly behind LSTM in terms of test loss.
- The **Feedforward Neural Network**, while decent, does not outperform the recurrent models.

---

## Code Details

### Preprocessing
1. **Sorting**: The dataset is sorted chronologically by date.
2. **Lag Features**: Features `gold_val_lag1` to `gold_val_lag5` were created.
3. **Normalization**: StandardScaler was used to normalize features.
4. **Train-Test Split**: 75% of the data was used for training, and 25% was reserved for testing.
5. **Batching**: Data was converted to PyTorch tensors and DataLoaders were created for training and testing.


---


## Conclusion
Recurrent neural networks, specifically LSTM and GRU, provide better performance for time-series forecasting compared to Feedforward Neural Networks. With only 5 LSTM units, the LSTM model achieves the best performance, demonstrating its ability to capture temporal patterns in time-series data.

---

## How to run the code
1. Clone the repository:
```bash
git clone https://github.com/your-username/gold-rate-prediction-lstm.git
```
2. Navigate to the project directory:
```bash
cd gold-rate-prediction-lstm
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
4. Run the code:
```bash
python gold.py
```
