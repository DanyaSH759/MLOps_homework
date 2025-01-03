import json
from datetime import datetime
from math import sqrt

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from train import (  # Импорт модели и функций из train.py
    CryptoPricePredictor,
    preprocess_time_series_data,
)


def main():

    with open("scripts/conf_train.json", "r") as f:
        config = json.load(f)

    filepath = config["filepath"]
    seq_length = config["seq_length"]
    batch_size = config["batch_size"]

    with open("scripts/conf_pred.json", "r") as f:
        config = json.load(f)

    checkpoint_path = config["checkpoint_path"]

    # Prepare data
    _, _, test_dataset, _ = preprocess_time_series_data(filepath, seq_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load model
    model = CryptoPricePredictor.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Predictions
    predictions = []
    targets = []

    for x, y in test_loader:
        with torch.no_grad():
            y_hat = model(x)
            predictions.extend(y_hat.squeeze().tolist())
            targets.extend(y.tolist())

    print(np.array(predictions).shape)
    print(np.array(targets).shape)

    mse = mean_squared_error(targets, predictions)
    rmse = sqrt(mse)
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    # Save predictions
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = pd.DataFrame({"Actual": targets, "Predicted": predictions})
    results.to_csv(f"predict/predictions_{current_date}.csv", index=False)
    print("Predictions saved to predictions.csv")


if __name__ == "__main__":
    main()
