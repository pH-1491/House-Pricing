import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess():
    df = pd.read_csv("data/housing.csv")

    df = df.drop("Address", axis=1)

    X = df.drop("Price", axis=1)
    y = df["Price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "model/scaler.pkl")

    return X_scaled, y.values