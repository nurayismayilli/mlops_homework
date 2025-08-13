import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_model(X_train, y_train, preprocessor):
    """Train logistic regression model with preprocessing."""
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))]
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filepath):
    """Save model to file."""
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load model from file."""
    return joblib.load(filepath)
