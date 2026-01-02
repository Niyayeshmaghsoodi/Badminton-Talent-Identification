import joblib
import pandas as pd

model = joblib.load("models/text_model.pkl")

sample = pd.DataFrame([{
    "age": 19,
    "height": 170,
    "weight": 62,
    "agility": 9,
    "endurance": 7,
    "hand_eye_coord": 8,
    "flexibility": 8
}])

print(model.predict(sample))
