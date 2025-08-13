from src.mlops_homework.features.preprocessing import load_and_preprocess_data
from src.mlops_homework.models.modeling import save_model, train_model

X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(
    "data/train.csv", "target"
)
model = train_model(X_train, y_train, preprocessor)
save_model(model, "models/model.joblib")
