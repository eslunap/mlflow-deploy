import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature


DATA_PATH = "data/winequality-red.csv"
MODEL_DIR = "model"
METRICS_DIR = "metrics"
MLRUNS_DIR = "mlruns"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("wine-quality-regression")


def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    X = df.drop(columns=["quality"])
    y = df["quality"]
    return X, y


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_path = os.path.join(MODEL_DIR, "wine_quality_model.pkl")
    joblib.dump(model, model_path)

    test_data_path = os.path.join(MODEL_DIR, "test_data.csv")
    test_df = X_test.copy()
    test_df["quality"] = y_test.values
    test_df.to_csv(test_data_path, index=False)

    metrics_path = os.path.join(METRICS_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(f"MSE: {mse:.4f}\n")
        file.write(f"R2: {r2:.4f}\n")

    input_example = X_train.head(3)
    signature = infer_signature(X_train, model.predict(X_train))

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("dataset", "Wine Quality Red")
        mlflow.log_param("target", "quality")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(test_data_path)
        mlflow.log_artifact(metrics_path)

        with open("run_id.txt", "w", encoding="utf-8") as file:
            file.write(run.info.run_id)

    print("Entrenamiento finalizado correctamente.")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")


if __name__ == "__main__":
    main()