import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    return df

def clean_data(df):
    df = df.dropna()
    df = df.sort_values('Date')
    return df

def add_features(df):
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df
