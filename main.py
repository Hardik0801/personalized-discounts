from pathlib import Path

import pandas as pd

from src.decision import recommend_discount
from src.models import load_model, save_model, train_model
from src.preprocess import (
    apply_user_values,
    build_transaction_template,
    load_data,
    preprocess_data,
    preprocess_new,
    show_basic_info,
    show_missing_values,
)

print("Personalized Discounts Project Started")


def find_data_file():
    default_path = Path("data/raw/ecommerce_behavior.csv")
    if default_path.exists():
        return default_path

    csv_files = sorted(Path("data/raw").glob("*.csv"))
    if csv_files:
        return csv_files[0]

    return None


def parse_user_value(raw_value, field_name):
    if raw_value is None or str(raw_value).strip() == "":
        return None

    numeric_fields = {
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
    }

    if field_name in numeric_fields:
        try:
            return float(raw_value)
        except ValueError:
            return raw_value

    return raw_value


if __name__ == "__main__":
    data_path = find_data_file()

    if data_path is None:
        print("Dataset not found. Please place your CSV file in data/raw")
    else:
        df = load_data(str(data_path), nrows=10000)
        show_basic_info(df)
        show_missing_values(df)

        X, y, numeric_medians, categorical_maps = preprocess_data(df)
        print("Features shape:", X.shape)
        print("Target shape:", y.shape)

        model = train_model(X, y)
        save_model(model)
        print("Model saved to models/fraud_model.joblib")

        choice = input("Do you want to enter a custom transaction? (y/n): ").strip().lower()

        if choice.startswith("y"):
            fields_to_prompt = [
                "TransactionDT",
                "TransactionAmt",
                "ProductCD",
                "card1",
                "card2",
                "card3",
                "card4",
                "card5",
                "card6",
                "addr1",
                "addr2",
                "dist1",
                "dist2",
                "P_emaildomain",
                "R_emaildomain",
            ]

            transaction_df = build_transaction_template(X.columns)
            values = {}

            print("Enter values for the fields below. Press Enter to leave blank.")
            for field in fields_to_prompt:
                if field not in X.columns:
                    continue

                raw_value = input(f"{field}: ").strip()
                parsed_value = parse_user_value(raw_value, field)
                if parsed_value is not None:
                    values[field] = parsed_value

            transaction_df = apply_user_values(transaction_df, values)
            sample_X = preprocess_new(transaction_df, X.columns, numeric_medians, categorical_maps)

            loaded_model = load_model()
            prob = loaded_model.predict_proba(sample_X)[0][1]
            discount = recommend_discount(prob)

            print("Fraud probability:", prob)
            print("Recommended discount:", f"{discount * 100:.1f}%")
        else:
            sample = df.iloc[[0]].copy()
            sample = sample.drop(columns=["isFraud"], errors="ignore")
            sample_X = preprocess_new(sample, X.columns, numeric_medians, categorical_maps)

            loaded_model = load_model()
            prob = loaded_model.predict_proba(sample_X)[0][1]
            discount = recommend_discount(prob)

            print("Fraud probability:", prob)
            print("Recommended discount:", f"{discount * 100:.1f}%")
