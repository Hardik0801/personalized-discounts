import pandas as pd

def load_data(path, nrows=None):
    return pd.read_csv(path, nrows=nrows)

def show_basic_info(df):
    print("Shape:", df.shape)
    print(df.head())
    print(df.columns.tolist())


def show_missing_values(df):
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if missing_counts.empty:
        print("No missing values found.")
        return

    print("Missing values by column:")
    print(missing_counts)


def preprocess_data(df, target_column="isFraud"):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    features = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in features.columns if col not in numeric_cols]

    numeric_medians = {}
    for col in numeric_cols:
        median_value = features[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        numeric_medians[col] = float(median_value)
        features[col] = features[col].fillna(median_value)

    categorical_maps = {}
    for col in categorical_cols:
        encoded = features[col].astype(str).fillna("__missing__")
        categories = sorted(encoded.unique())
        mapping = {value: idx for idx, value in enumerate(categories)}
        categorical_maps[col] = mapping
        features[col] = encoded.map(mapping).astype(float)

    X = features.astype(float)
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    return X, y, numeric_medians, categorical_maps


def build_transaction_template(columns):
    return pd.DataFrame([{col: None for col in columns}])


def apply_user_values(transaction_df, values):
    updated = transaction_df.copy()
    for key, value in values.items():
        if key in updated.columns:
            updated.at[updated.index[0], key] = value
    return updated


def preprocess_new(df_new, train_columns, numeric_medians, categorical_maps):
    processed = df_new.copy()

    for col in train_columns:
        if col not in processed.columns:
            processed[col] = None

    processed = processed[list(train_columns)]

    for col, median_value in numeric_medians.items():
        if col in processed.columns:
            processed[col] = pd.to_numeric(processed[col], errors="coerce").fillna(median_value)

    for col, mapping in categorical_maps.items():
        if col in processed.columns:
            encoded = processed[col].astype(str).fillna("__missing__")
            processed[col] = encoded.map(mapping).fillna(-1).astype(float)

    return processed.astype(float)