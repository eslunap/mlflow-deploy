import os
import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error, r2_score


MLRUNS_DIR = "mlruns"
RUN_ID_PATH = "run_id.txt"
TEST_DATA_PATH = "model/test_data.csv"

MSE_THRESHOLD = 0.50


mlflow.set_tracking_uri("file:./mlruns")


def main():
    if not os.path.exists(RUN_ID_PATH):
        raise FileNotFoundError("No se encontró run_id.txt. Ejecuta primero train.py.")

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError("No se encontró model/test_data.csv. Ejecuta primero train.py.")

    with open(RUN_ID_PATH, "r", encoding="utf-8") as file:
        run_id = file.read().strip()

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=["quality"])
    y_test = test_df["quality"]

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Modelo cargado desde MLflow: {model_uri}")
    print(f"MSE obtenido: {mse:.4f}")
    print(f"R2 obtenido: {r2:.4f}")
    print(f"Umbral máximo permitido de MSE: {MSE_THRESHOLD}")

    if mse <= MSE_THRESHOLD:
        print("Validación exitosa: el modelo cumple el umbral definido.")
    else:
        raise ValueError(
            f"Validación fallida: el MSE ({mse:.4f}) supera el umbral ({MSE_THRESHOLD})."
        )


if __name__ == "__main__":
    main()