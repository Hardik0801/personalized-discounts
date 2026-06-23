from src.preprocess import load_data, show_basic_info

print("Personalized Discounts Project Started")

# Example usage
df = load_data("data/raw/ecommerce_behavior.csv")
show_basic_info(df)