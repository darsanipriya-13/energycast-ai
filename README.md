# ⚡ EnergyCast AI — LSTM-Based Energy Forecasting

An AI-powered web application that analyzes historical energy data and forecasts the next year's energy value using a Long Short-Term Memory (LSTM) neural network.

🔗 **Live Demo:** https://energycast-ai.onrender.com/

📂 **GitHub Repository:** https://github.com/darsanipriya-13/energycast-ai

---

## 📌 Project Overview

EnergyCast AI is a machine learning web application developed to explore historical energy trends and predict future energy values from yearly data.

The application uses an LSTM model, a type of recurrent neural network designed to learn patterns from sequential data. It takes the previous five years of energy values as input to generate a next-year forecast.

The project combines time-series forecasting, data visualization, model evaluation, and interactive web deployment in a single application.

## 🎯 Objectives

- Analyze historical yearly energy data.
- Identify patterns in time-series energy values.
- Forecast the next year's energy value using an LSTM model.
- Evaluate predictions using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Provide an interactive interface for exploring country-level forecasts.
- Make the application accessible through a publicly hosted web link.

## ✨ Key Features

- **Country Selection:** Select a country from the available dataset.
- **Historical Data Visualization:** View yearly energy values in a line chart.
- **LSTM Forecasting:** Predict the next year's energy value using the previous five years.
- **Model Evaluation:** Compare actual and predicted values for the test period.
- **Performance Metrics:** Display MAE and RMSE.
- **CSV Report Download:** Download historical values and the forecast as a CSV file.
- **Interactive Web Interface:** Use the application through a Gradio interface.
- **Cloud Deployment:** Access the application online through Render.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Data normalization and evaluation metrics |
| TensorFlow / Keras | LSTM model development and training |
| Matplotlib | Data visualization |
| Gradio | Interactive web interface |
| Render | Cloud hosting and deployment |
| Git & GitHub | Version control and source code management |

## ⚙️ How It Works

1. **Data Loading:** Load a CSV dataset containing country names and yearly energy values.
2. **Data Preprocessing:** Convert yearly values to numeric data, handle missing values, and select valid historical observations.
3. **Normalization:** Scale the energy values using MinMaxScaler.
4. **Sequence Creation:** Create input sequences using the previous five years of data.
5. **Model Training:** Train an LSTM neural network on earlier observations.
6. **Model Evaluation:** Evaluate predictions on a later, chronologically separated test set using MAE and RMSE.
7. **Forecast Generation:** Use the latest five yearly values to predict the next year.
8. **Visualization and Reporting:** Display the forecast, compare test predictions, and provide a downloadable CSV report.

## 🏗️ System Workflow

```text
Historical Energy Dataset (CSV)
              |
              v
       Data Preprocessing
              |
              v
      Missing Value Handling
              |
              v
       MinMax Normalization
              |
              v
    Five-Year Sequence Creation
              |
              v
        LSTM Model Training
              |
              v
       Model Evaluation
         (MAE and RMSE)
              |
              v
       Next-Year Forecast
              |
              v
     Gradio Web Application
              |
              v
      Render Cloud Deployment
```

## 📊 Model Evaluation

EnergyCast AI uses two regression metrics to evaluate its test predictions.

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.
- **Root Mean Squared Error (RMSE):** Measures prediction error while giving greater weight to larger errors.

Lower values generally indicate smaller prediction errors when comparing models on the same dataset and scale.

## 📁 Project Structure

```text
energycast-ai/
│
├── app.py
├── requirements.txt
├── energy_data.csv
└── README.md
```

*Note: `energy_data.csv` is an example filename. Use the actual CSV filename in your repository.*

## 🚀 Run Locally

### Prerequisites

- Python 3.10 or another Python version compatible with your installed TensorFlow release
- pip
- Git

### 1. Clone the repository

```bash
git clone https://energycast-ai.onrender.com/
```

### 2. Navigate to the project folder

```bash
cd energycast-ai
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the application

```bash
python app.py
```

Open the local URL displayed in the terminal to use the application.

## ☁️ Deployment

EnergyCast AI is deployed using **Render**, making the application accessible through a web browser without requiring users to run the code locally.

- **Live Application:** https://energycast-ai.onrender.com/
- **Source Code:** https://github.com/darsanipriya-13/energycast-ai

The application uses a Python web service with Gradio and TensorFlow CPU dependencies.

## 🔮 Future Enhancements

- Compare LSTM with GRU and other time-series forecasting models.
- Add forecasting for multiple future years.
- Improve missing-value handling and data validation.
- Add more model evaluation and visualization options.
- Introduce automated model comparison and hyperparameter tuning.
- Integrate additional energy datasets and more detailed trend analysis.

## 🎓 Learning Outcomes

Through this project, I gained practical experience with:

- Time-series data preprocessing and normalization.
- Sequential data preparation for deep learning.
- LSTM model training and prediction.
- Regression model evaluation.
- Interactive machine learning application development.
- GitHub version control and cloud deployment.

## ⚠️ Disclaimer

The forecast is an estimate based on historical data and the trained model. It is not a guarantee of future energy consumption. Prediction quality depends on data quality, the amount of available historical information, and model performance.

## 👩‍💻 Author

**Darsani Priya RJ**

Computer Science and Engineering Student

---

⭐ If you find this project useful, consider starring the repository!

