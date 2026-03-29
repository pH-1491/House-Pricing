from fastapi import FastAPI
from app.schemas import HouseInput
from app.predict import predict_price
import os

port = int(os.environ.get("PORT", 8000))
app = FastAPI()

@app.get("/")
def home():
    return {"message": "House Price API running"}

@app.post("/predict")
def predict(input_data: HouseInput):
    price = predict_price(input_data.data)
    return {"predicted_price": price}