# ⚡ Electricity Demand Forecasting Using Deep Learning

A deep learning-based **electricity demand forecasting** project using a hybrid **Bidirectional LSTM and Convolutional Neural Network (CNN)** architecture to predict electricity demand in India from historical demand data.

## 📌 Project Overview

Electricity demand forecasting is an important time-series problem for energy planning, grid management, and resource allocation.

This project uses historical **All India electricity demand data** and develops a deep learning model capable of learning temporal patterns from previous demand observations.

The model uses the previous **60 observations** to predict the next electricity-demand value. A hybrid architecture combining **Bidirectional LSTM, Conv1D, MaxPooling and Dense layers** is used to capture sequential and local patterns in the demand data.

The complete implementation is provided in:

`Final_Project_LSTM.ipynb`

---

## 🎯 Objectives

* Analyze historical electricity demand data for India.
* Prepare and transform the dataset for time-series forecasting.
* Normalize demand values using Min-Max scaling.
* Create sequential input windows using the previous 60 observations.
* Build a hybrid Bidirectional LSTM-CNN deep learning model.
* Train the model to forecast electricity demand.
* Generate predictions for the final 1,000 observations.
* Compare actual and predicted electricity demand.
* Calculate demand deviation and percentage deviation.

---

## 🗂️ Dataset

The project uses historical electricity demand data for **All India**.

The dataset is stored as:

```text
demand-for-all-india-fro.csv
```

The original dataset contains a `Category` field from which the notebook extracts:

* **Date**
* **Demand Value**

The extracted date is converted into a Pandas datetime format and the observations are sorted chronologically before modelling.

### Target Variable

```text
All India
```

The `All India` electricity-demand series is used as the target variable for forecasting.

---

## 🔄 Project Workflow

```text
Historical Electricity Demand Data
                │
                ▼
        Data Preprocessing
                │
                ▼
      Extract Date & Demand
                │
                ▼
       Sort by Date
                │
                ▼
        Min-Max Scaling
                │
                ▼
   Create 60-Step Sequences
                │
                ▼
     Bidirectional LSTM
                │
                ▼
           Dropout
                │
                ▼
           Conv1D
                │
                ▼
        Max Pooling
                │
                ▼
            Flatten
                │
                ▼
        Dense Layer
                │
                ▼
      Predicted Demand
                │
                ▼
      Inverse Scaling
                │
                ▼
Actual vs Predicted Demand
```

---

## 🧹 Data Preprocessing

The notebook performs the following preprocessing steps:

### 1. Load the dataset

```python
df = pd.read_csv('demand-for-all-india-fro.csv')
```

### 2. Extract the date

The date is extracted from the `Category` column and converted into a datetime object.

### 3. Extract the demand value

The demand value is extracted from the `Category` field and converted to an integer.

### 4. Remove unnecessary columns

The intermediate `Category` and `Value` columns are removed after the required information has been extracted.

### 5. Sort chronologically

The dataset is sorted by `Date` to preserve the time-series sequence.

### 6. Normalize demand

The `All India` demand values are scaled to the range **0–1** using:

```python
MinMaxScaler(feature_range=(0, 1))
```

This helps the neural network train more efficiently.

---

## 🧠 Time-Series Sequence Creation

The model uses a **60-observation lookback window**.

For every prediction:

```text
Previous 60 observations
          ↓
       LSTM-CNN
          ↓
Next demand value
```

For example:

```text
t1, t2, t3, ..., t60  →  t61
t2, t3, t4, ..., t61  →  t62
t3, t4, t5, ..., t62  →  t63
```

The resulting input is reshaped into the three-dimensional format required by the neural network:

```text
(samples, time steps, features)
```

with:

```text
time steps = 60
features = 1
```

---

# 🧠 Model Architecture

The project uses a hybrid **Bidirectional LSTM + CNN** architecture.

### Architecture

```text
Input
  │
  │ 60 time steps × 1 feature
  ▼
Bidirectional LSTM
50 units
return_sequences=True
  │
  ▼
Dropout
20%
  │
  ▼
Conv1D
64 filters
kernel size = 3
ReLU activation
  │
  ▼
MaxPooling1D
pool size = 2
  │
  ▼
Flatten
  │
  ▼
Dense
50 neurons
ReLU activation
  │
  ▼
Dense
1 neuron
  │
  ▼
Predicted Electricity Demand
```

### Why this architecture?

**Bidirectional LSTM**

Captures temporal relationships within the input sequence by processing the sequence in both directions.

**Dropout**

A dropout rate of **20%** is used to reduce the risk of overfitting.

**Conv1D**

The convolutional layer helps identify local patterns within the sequential demand data.

**MaxPooling1D**

Reduces the dimensionality of the extracted features while retaining important information.

**Dense layers**

Transform the extracted features into the final electricity-demand prediction.

---

## ⚙️ Model Configuration

| Parameter          |              Value |
| ------------------ | -----------------: |
| Lookback Window    |    60 observations |
| LSTM Type          | Bidirectional LSTM |
| LSTM Units         |                 50 |
| Dropout            |               0.20 |
| Conv1D Filters     |                 64 |
| Conv1D Kernel Size |                  3 |
| Pool Size          |                  2 |
| Dense Units        |                 50 |
| Output Units       |                  1 |
| Optimizer          |               Adam |
| Loss Function      | Mean Squared Error |
| Epochs             |                 50 |
| Batch Size         |                 32 |
| Total Parameters   |            132,965 |

The Keras model summary reports **132,965 trainable parameters**.

---

# 📈 Model Training

The model was trained for **50 epochs** using:

```python
regressor.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32
)
```

The training loss decreased consistently throughout training:

| Epoch | Training Loss |
| ----: | ------------: |
|     1 |        0.0077 |
|     5 |        0.0044 |
|    10 |        0.0032 |
|    20 |        0.0020 |
|    30 |        0.0016 |
|    40 |        0.0014 |
|    50 |        0.0012 |

This represents a substantial reduction in training loss during the 50 training epochs.

> **Note:** The reported loss is Mean Squared Error on the **scaled demand values**, so it should not be interpreted directly as MW².

---

# 🔮 Forecasting

For evaluation, the notebook uses the **last 1,000 observations** as the testing set.

The previous 60 observations are included when constructing the input sequences so that the model has the required historical context for each prediction.

The predicted values are then converted back from the scaled representation to their original demand scale using:

```python
scaler.inverse_transform(predicted_stock_price)
```

---

# 📊 Prediction Analysis

The notebook creates a comparison DataFrame containing:

```text
Date
True_Demand
Predicted_Demand
Demand_Deviation
Demand_Deviation_Percentage
```

### Demand Deviation

The absolute difference between actual and predicted demand is calculated as:

```python
abs(True_Demand - Predicted_Demand)
```

### Demand Deviation Percentage

The percentage deviation is calculated as:

```python
(abs(True_Demand - Predicted_Demand) / True_Demand) * 100
```

This allows the forecasting error to be examined both in absolute demand units and relative percentage terms.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* MinMaxScaler

### Deep Learning

* TensorFlow
* Keras

### Model Components

* Bidirectional LSTM
* Dropout
* Conv1D
* MaxPooling1D
* Flatten
* Dense

---

## 📁 Repository Structure

```text
Electricity_Demand_Forecasting/
│
├── Final_Project_LSTM.ipynb
│   └── Complete forecasting implementation
│
├── demand-for-all-india-fro.csv
│   └── Historical India electricity-demand dataset
│
└── README.md
    └── Project documentation
```

The repository contains the forecasting notebook and associated dataset.

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Shreyas6Naik/Electricity_Demand_Forecasting.git
```

### 2. Navigate to the project directory

```bash
cd Electricity_Demand_Forecasting
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
Final_Project_LSTM.ipynb
```

Run the notebook cells sequentially.

---

## 💡 Key Takeaways

* Electricity demand can be modelled effectively as a **time-series forecasting problem**.
* A **60-step historical window** was used to generate predictions.
* The hybrid Bidirectional LSTM-CNN architecture combines sequential learning with convolutional feature extraction.
* Training loss decreased from **0.0077 to 0.0012** over 50 epochs.
* The model generated predictions for the final **1,000 observations** and calculated both absolute and percentage demand deviations.

---

## 🔮 Future Improvements

The project can be extended by:

* Evaluating the model using **MAE, RMSE and MAPE**.
* Creating a separate validation set instead of training exclusively on the training observations.
* Comparing the hybrid model against standard **LSTM, GRU, ARIMA and XGBoost** models.
* Incorporating external variables such as temperature, weather, holidays and seasonality.
* Performing hyperparameter tuning.
* Implementing multi-step forecasting.
* Deploying the forecasting model through a Streamlit application.
* Building an interactive dashboard for electricity-demand monitoring.

---

## 👨‍💻 Author

**Shreyas Naik**

GitHub:
https://github.com/Shreyas6Naik

---

## ⭐ Project Summary

**Project:** Electricity Demand Forecasting
**Domain:** Energy Analytics
**Problem:** Time-Series Forecasting
**Model:** Bidirectional LSTM + CNN
**Framework:** TensorFlow / Keras
**Language:** Python
**Target:** All India Electricity Demand
