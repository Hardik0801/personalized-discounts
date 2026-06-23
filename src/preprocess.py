import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def show_basic_info(df):
    print("Shape:", df.shape)
    print(df.head())
    print(df.columns.tolist())