import torch
import joblib
from app.model import HouseModel

model = HouseModel()
model.load_state_dict(torch.load("model/model.pth"))
model.eval()

scaler = joblib.load("model/scaler.pkl")

def predict_price(data):
    scaled = scaler.transform(data)
    x = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        pred = model(x).item()

    return pred