import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

X = np.array([[0.8, 0.7], [0.4, 0.5], [0.9, 0.8]])
y = [1, 0, 1]  # win / lose

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "models/match_predictor.pkl")
