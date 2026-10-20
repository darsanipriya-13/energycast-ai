!pip -q install gradio scikit-learn tensorflow matplotlib

import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# 1. Use the existing dataset
if "df" not in globals():
    raise ValueError("Run your dataset upload cell first!")

df["Country"] = df["Country"].astype(str).str.strip()

YEAR_COLS = sorted(
    [c for c in df.columns if str(c).strip().isdigit()],
    key=lambda c: int(str(c).strip())
)

if len(YEAR_COLS) < 12:
    raise ValueError("Year columns not detected correctly.")

COUNTRIES = sorted(
    c for c in df["Country"].dropna().unique()
    if c and c.lower() != "nan"
)

LOOKBACK = 5
EPOCHS = 40


# 2. Prepare selected country's data
def prepare_data(country_name):

    country_name = str(country_name or "").strip()

    if not country_name:
        raise gr.Error("Please enter a country name.")

    matches = [
        c for c in COUNTRIES
        if c.lower() == country_name.lower()
    ]

    if not matches:
        suggestions = [
            c for c in COUNTRIES
            if country_name.lower() in c.lower()
        ][:8]

        message = "Country not found."
        if suggestions:
            message += " Suggestions: " + ", ".join(suggestions)

        raise gr.Error(message)

    country = matches[0]

    rows = df[
        df["Country"].str.lower() == country.lower()
    ]

    # Handles duplicate country rows by averaging yearly values
    numeric_rows = rows[YEAR_COLS].apply(
        pd.to_numeric, errors="coerce"
    )

    energy = numeric_rows.mean(axis=0, skipna=True)

    data = pd.DataFrame({
        "Year": [int(str(c)) for c in YEAR_COLS],
        "Energy": energy.to_numpy(dtype=float)
    })

    data = data.sort_values("Year").reset_index(drop=True)

    data["Energy"] = data["Energy"].interpolate(
        limit_direction="both"
    )

    data = data.dropna(subset=["Energy"]).reset_index(drop=True)

    if len(data) < 15:
        raise gr.Error(
            "Not enough yearly records. Choose another country."
        )

    return country, data


# 3. Train model and generate forecast
def generate_forecast(country_name):

    country, data = prepare_data(country_name)

    values = data["Energy"].to_numpy().reshape(-1, 1)

    split = int(len(data) * 0.8)

    if split <= LOOKBACK or split >= len(data):
        raise gr.Error("Not enough data for training and testing.")

    # Fit scaler using training data only
    scaler = MinMaxScaler()
    scaler.fit(values[:split])

    scaled = scaler.transform(values)

    X, y, target_indices = [], [], []

    for i in range(LOOKBACK, len(data)):
        X.append(scaled[i-LOOKBACK:i, 0])
        y.append(scaled[i, 0])
        target_indices.append(i)

    X = np.array(X, dtype=np.float32).reshape(
        -1, LOOKBACK, 1
    )

    y = np.array(y, dtype=np.float32)
    target_indices = np.array(target_indices)

    train_mask = target_indices < split
    test_mask = target_indices >= split

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    test_indices = target_indices[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise gr.Error("Insufficient training or testing records.")

    # LSTM model
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(32, activation="tanh"),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=min(8, len(X_train)),
        shuffle=False,
        verbose=0
    )

    # 4. Test predictions
    predicted_scaled = model.predict(X_test, verbose=0)

    actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    predicted = scaler.inverse_transform(
        predicted_scaled
    ).flatten()

    # Simple baseline: use previous year's actual value
    baseline_scaled = scaled[
        test_indices - 1, 0
    ].reshape(-1, 1)

    baseline = scaler.inverse_transform(
        baseline_scaled
    ).flatten()

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    baseline_mae = mean_absolute_error(actual, baseline)

    baseline_rmse = np.sqrt(
        mean_squared_error(actual, baseline)
    )

    # 5. Forecast next year
    last_five = scaled[-LOOKBACK:].reshape(
        1, LOOKBACK, 1
    )

    next_scaled = model.predict(last_five, verbose=0)

    next_energy = float(
        scaler.inverse_transform(next_scaled)[0, 0]
    )

    last_year = int(data["Year"].iloc[-1])
    next_year = last_year + 1

    # 6. Historical graph
    fig1, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        data["Year"], data["Energy"],
        marker="o", markersize=3,
        label="Historical consumption"
    )

    ax.set_title(f"Energy Consumption - {country}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Energy consumption (dataset units)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig1.tight_layout()

    # 7. Actual vs predicted graph
    test_years = data["Year"].to_numpy()[test_indices]

    fig2, ax = plt.subplots(figsize=(10, 4))

    ax.plot(test_years, actual, marker="o", label="Actual")
    ax.plot(test_years, predicted, marker="x", label="LSTM")
    ax.plot(
        test_years, baseline,
        linestyle="--", label="Previous-year baseline"
    )

    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Year")
    ax.set_ylabel("Energy consumption (dataset units)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig2.tight_layout()

    # 8. Results table and CSV report
    results = pd.DataFrame({
        "Country": country,
        "Year": test_years,
        "Actual": actual,
        "LSTM_Predicted": predicted,
        "Previous_Year_Baseline": baseline,
        "Absolute_Error": np.abs(actual - predicted)
    })

    forecast_row = pd.DataFrame([{
        "Country": country,
        "Year": next_year,
        "Actual": np.nan,
        "LSTM_Predicted": next_energy,
        "Previous_Year_Baseline": float(data["Energy"].iloc[-1]),
        "Absolute_Error": np.nan
    }])

    report = pd.concat(
        [results, forecast_row],
        ignore_index=True
    )

    safe_name = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in country
    )

    report_path = os.path.join(
        tempfile.gettempdir(),
        f"EnergyCast_AI_{safe_name}_report.csv"
    )

    report.to_csv(report_path, index=False)

    summary = f"""
# ⚡ EnergyCast AI — Forecast Results

## Country: {country}

| Prediction | Result |
|---|---:|
| Latest available year | {last_year} |
| Forecast year | **{next_year}** |
| Predicted consumption | **{next_energy:,.4f} dataset units** |

## Model Evaluation

| Metric | LSTM | Previous-year baseline |
|---|---:|---:|
| MAE | {mae:,.4f} | {baseline_mae:,.4f} |
| RMSE | {rmse:,.4f} | {baseline_rmse:,.4f} |

Training sequences: {len(X_train)}

Testing sequences: {len(X_test)}

Input window: Previous {LOOKBACK} years

*Forecasts are estimates, not guarantees. Missing values are interpolated.
Compare the LSTM results with the baseline before interpreting performance.*
"""

    return summary, fig1, fig2, results, report_path


# 9. Website interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="EnergyCast AI"
) as app:

    gr.Markdown("""
    # ⚡ EnergyCast AI
    ### World Energy Consumption Forecasting Platform

    Enter a country name to explore historical energy consumption
    and forecast the next year using an LSTM model.
    """)

    with gr.Row():

        country_input = gr.Dropdown(
            choices=COUNTRIES,
            value=(
                "India"
                if any(c.lower() == "india" for c in COUNTRIES)
                else COUNTRIES[0]
            ),
            label="Search country",
            info="Type to filter countries and select a suggestion.",
            allow_custom_value=True,
            interactive=True
        )

        predict_button = gr.Button(
            "Generate Forecast",
            variant="primary"
        )

    summary_output = gr.Markdown()

    with gr.Tabs():

        with gr.Tab("Historical Analytics"):
            history_plot = gr.Plot()

        with gr.Tab("Model Validation"):
            validation_plot = gr.Plot()

        with gr.Tab("Prediction Data"):
            results_table = gr.Dataframe(
                interactive=False
            )

        with gr.Tab("Download Report"):
            download_file = gr.File(
                label="Download CSV report"
            )

    predict_button.click(
        fn=generate_forecast,
        inputs=country_input,
        outputs=[
            summary_output,
            history_plot,
            validation_plot,
            results_table,
            download_file
        ]
    )

print("EnergyCast AI website is starting...")

app.launch(share=True, debug=True)
