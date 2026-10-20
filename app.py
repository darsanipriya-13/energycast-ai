
import os
import glob
import tempfile

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

APP_TITLE = "EnergyCast AI"
LOOKBACK = 5
EPOCHS = 20

# Find a CSV file in the project folder
def load_dataset():
    possible_files = (
        glob.glob("*.csv")
        + glob.glob("data/*.csv")
        + glob.glob("dataset/*.csv")
    )

    for file in possible_files:
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.astype(str).str.strip()

            if "Country" in df.columns:
                return df, file
        except Exception:
            continue

    raise FileNotFoundError(
        "CSV dataset not found. Upload your energy dataset "
        "with a Country column beside app.py."
    )


def get_year_columns(df):
    return sorted(
        [c for c in df.columns if c.isdigit()],
        key=int
    )


def get_country_data(country):
    df, filename = load_dataset()
    year_cols = get_year_columns(df)

    if not year_cols:
        raise ValueError(
            "No year columns found. Expected columns such as 1965, 1966, 1967."
        )

    country_rows = df[
        df["Country"].astype(str).str.strip() == str(country).strip()
    ]

    if country_rows.empty:
        raise ValueError("Selected country was not found in the dataset.")

    row = country_rows.iloc[0]

    values = pd.to_numeric(
        row[year_cols], errors="coerce"
    )

    values.index = pd.to_numeric(values.index)
    values = values.sort_index()
    values = values.dropna()

    if len(values) < LOOKBACK + 2:
        raise ValueError(
            f"Not enough valid yearly values for {country}. "
            f"At least {LOOKBACK + 2} years are required."
        )

    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains invalid numeric values.")

    return values, filename


# --------------------------------------------------
# FORECAST FUNCTION
# --------------------------------------------------

def generate_forecast(country):
    try:
        values, filename = get_country_data(country)

        years = values.index.to_numpy(dtype=int)
        energy = values.to_numpy(dtype=float)

        # Scale the energy values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(
            energy.reshape(-1, 1)
        )

        # Create sequences using the previous 5 years
        X, y = [], []

        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i - LOOKBACK:i, 0])
            y.append(scaled[i, 0])

        X = np.array(X).reshape(-1, LOOKBACK, 1)
        y = np.array(y)

        if len(X) < 2:
            raise ValueError("Not enough data to train and evaluate the model.")

        # Chronological train/test split
        split = max(1, int(len(X) * 0.8))
        split = min(split, len(X) - 1)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Import TensorFlow only when a forecast is requested
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, LSTM, Dense
        from tensorflow.keras.callbacks import EarlyStopping

        tf.keras.utils.set_random_seed(42)

        model = Sequential([
            Input(shape=(LOOKBACK, 1)),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1)
        ])

        model.compile(
            optimizer="adam",
            loss="mean_squared_error"
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=min(8, len(X_train)),
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )

        # Evaluate on later years
        predictions = model.predict(X_test, verbose=0)

        actual_values = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        predicted_values = scaler.inverse_transform(
            predictions
        ).flatten()

        mae = mean_absolute_error(
            actual_values, predicted_values
        )

        rmse = np.sqrt(
            mean_squared_error(actual_values, predicted_values)
        )

        # Forecast the next year using the latest 5 years
        last_five = scaled[-LOOKBACK:].reshape(
            1, LOOKBACK, 1
        )

        next_scaled = model.predict(last_five, verbose=0)

        next_value = float(
            scaler.inverse_transform(next_scaled)[0, 0]
        )

        next_year = int(years[-1]) + 1

        # Historical chart + next-year forecast
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            years, energy,
            marker="o",
            markersize=3,
            label="Historical data"
        )

        ax.scatter(
            [next_year], [next_value],
            marker="*",
            s=180,
            label="LSTM forecast"
        )

        ax.plot(
            [years[-1], next_year],
            [energy[-1], next_value],
            linestyle="--"
        )

        ax.set_title(f"{country}: Energy Forecast")
        ax.set_xlabel("Year")
        ax.set_ylabel("Energy value (dataset units)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        # Evaluation chart
        eval_fig, eval_ax = plt.subplots(figsize=(9, 4))

        test_years = years[LOOKBACK + split:]

        eval_ax.plot(
            test_years[:len(actual_values)],
            actual_values,
            marker="o",
            label="Actual"
        )

        eval_ax.plot(
            test_years[:len(predicted_values)],
            predicted_values,
            marker="x",
            label="Predicted"
        )

        eval_ax.set_title("LSTM Test Results")
        eval_ax.set_xlabel("Year")
        eval_ax.set_ylabel("Energy value")
        eval_ax.grid(True, alpha=0.3)
        eval_ax.legend()
        eval_fig.tight_layout()

        # Downloadable report
        report = pd.DataFrame({
            "Year": list(years) + [next_year],
            "Energy_Value": list(energy) + [next_value],
            "Type": (
                ["Historical"] * len(years) + ["Forecast"]
            )
        })

        report_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="energycast_report_",
            delete=False,
            newline=""
        )

        report_path = report_file.name
        report_file.close()
        report.to_csv(report_path, index=False)

        summary = (
            f"### Forecast for {country}\n\n"
            f"- **Forecast year:** {next_year}\n"
            f"- **Predicted energy value:** {next_value:,.4f}\n"
            f"- **MAE:** {mae:,.4f}\n"
            f"- **RMSE:** {rmse:,.4f}\n"
            f"- **Latest historical year:** {years[-1]}\n\n"
            "MAE and RMSE are measured on the held-out test years. "
            "Lower values generally indicate smaller prediction errors."
        )

        return summary, fig, eval_fig, report_path

    except Exception as e:
        error = f"**Error:** {type(e).__name__}: {e}"
        return error, None, None, None


# --------------------------------------------------
# LOAD COUNTRIES
# --------------------------------------------------

try:
    dataset, dataset_file = load_dataset()

    countries = sorted(
        dataset["Country"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

except Exception as e:
    countries = []
    dataset_file = None
    startup_error = str(e)


# --------------------------------------------------
# GRADIO INTERFACE
# --------------------------------------------------

CUSTOM_CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
#main-title {
    text-align: center;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan"
    ),
    css=CUSTOM_CSS,
    title=APP_TITLE
) as app:

    gr.Markdown(
        """
        # ⚡ EnergyCast AI
        ### Energy Consumption Forecasting using LSTM

        Analyze historical yearly energy data and predict the
        next year's value using a Long Short-Term Memory model.
        """,
        elem_id="main-title"
    )

    if not countries:
        gr.Markdown(
            f"""
            ### Dataset loading error

            {startup_error}

            Upload the correct CSV file beside `app.py`.
            The dataset must contain a `Country` column and
            yearly columns such as `1965`, `1966`, `1967`.
            """
        )

    else:
        with gr.Row():
            country_input = gr.Dropdown(
                choices=countries,
                value=countries[0],
                label="Select Country",
                interactive=True
            )

            generate_btn = gr.Button(
                "Generate Forecast",
                variant="primary"
            )

        summary_output = gr.Markdown()

        with gr.Tab("Historical Data & Forecast"):
            forecast_plot = gr.Plot(
                label="Historical Data and Next-Year Forecast"
            )

        with gr.Tab("Model Evaluation"):
            evaluation_plot = gr.Plot(
                label="Actual vs Predicted Test Values"
            )

        report_output = gr.File(
            label="Download Forecast CSV Report"
        )

        generate_btn.click(
            fn=generate_forecast,
            inputs=[country_input],
            outputs=[
                summary_output,
                forecast_plot,
                evaluation_plot,
                report_output
            ]
        )

    gr.Markdown(
        """
        ---
        **Disclaimer:** Forecasts are estimates based on historical data.
        Results depend on data quality and model performance.
        """
    )


# --------------------------------------------------
# LAUNCH SERVER
# --------------------------------------------------

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
