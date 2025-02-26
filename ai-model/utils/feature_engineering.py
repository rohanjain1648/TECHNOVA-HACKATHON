import pandas as pd

def extract_features(transaction):
    # Example: Convert transaction data to feature vector
    features = pd.DataFrame([transaction])
    features = pd.get_dummies(features, columns=['transaction_type', 'location'], drop_first=True)
    return features
