def recommend_discount(fraud_probability):
    probability = max(0.0, min(1.0, float(fraud_probability)))

    if probability >= 0.8:
        return 0.0
    if probability >= 0.5:
        return 0.05
    if probability >= 0.3:
        return 0.1
    return 0.2
