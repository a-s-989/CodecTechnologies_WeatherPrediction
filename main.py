from src.data_preprocessing import load_data, clean_data, add_features
from src.model import train_model, predict, evaluate_model
from src.utils import plot_predictions
import numpy as np
import pandas as pd

def main():
    df = load_data('data/historical_weather.csv')
    df = clean_data(df)
    df = add_features(df)

    # Train/test split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    X_train = train_df[['DayOfYear']].values
    y_train = train_df['Temperature'].values
    X_test = test_df[['DayOfYear']].values
    y_test = test_df['Temperature'].values

    model = train_model(X_train, y_train)

    # Evaluate
    y_pred = predict(model, X_test)
    rmse, mae = evaluate_model(y_test, y_pred)
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")

    # Future predictions
    future_days = np.arange(df['DayOfYear'].max() + 1, df['DayOfYear'].max() + 31).reshape(-1, 1)
    preds = predict(model, future_days)
    future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=30)
    pred_df = pd.DataFrame({'Date': future_dates, 'PredictedTemp': preds})
    print(pred_df)

    # Plots
    plot_predictions(test_df['Date'], y_test, y_pred)

if __name__ == "__main__":
    main()
