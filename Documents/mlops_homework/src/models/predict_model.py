import pandas as pd

from src.mlops_homework.models.modeling import load_model

model = load_model("models/model.joblib")

new_data = pd.read_csv("data/new_data.csv")
predictions = model.predict(new_data)
print(predictions)
