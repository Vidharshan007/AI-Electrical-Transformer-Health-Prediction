import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model():
    data = pd.read_csv("transformer_data.csv")

    X = data.drop('Health', axis=1)
    y = data['Health']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model