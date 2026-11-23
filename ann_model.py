from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COLUMNS = ["Temperature(°C)", "Humidity(%)", "Wind speed (m/s)"]
LABELS = ["기온(°C)", "습도(%)", "풍속(m/s)"]
CSV_PATH = Path(__file__).with_name("SeoulBikeData_1000_3inputs.csv")


@lru_cache(maxsize=1)
def load_ann():
    data = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    features = data[COLUMNS]
    target = data["Rented Bike Count"]

    x_train, _, y_train, _ = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                max_iter=2000,
                early_stopping=True,
                random_state=42,
            ),
        ),
        transformer=StandardScaler(),
    )
    model.fit(x_train, y_train)
    limits = [(float(features[c].min()), float(features[c].max())) for c in COLUMNS]
    return model, limits


def predict_ann(temperature, humidity, wind_speed):
    model, limits = load_ann()
    values = [temperature, humidity, wind_speed]

    for label, value, (low, high) in zip(LABELS, values, limits):
        if not low <= value <= high:
            raise ValueError(f"{label}은 {low:g}~{high:g} 범위로 입력해 주세요.")

    features = pd.DataFrame([values], columns=COLUMNS)
    return round(max(0.0, float(model.predict(features)[0])))
