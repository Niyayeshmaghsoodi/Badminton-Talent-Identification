from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/text_model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    result = model.predict(df)
    return {"recommended_sport": result[0]}
