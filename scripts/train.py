import hydra
import mlflow
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class CryptoDataset(Dataset):

    """Класс датасета"""

    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length, :-1]
        y = self.data[idx + self.seq_length, -1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


# === LSTM Model ===
class CryptoPricePredictor(pl.LightningModule):

    """Класс нейросети"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, lr):
        super(CryptoPricePredictor, self).__init__()
        self.save_hyperparameters()  # Сохраняем гиперпараметры
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.lr = lr

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        return self.fc(hidden)

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(1))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def preprocess_time_series_data(file_path, seq_length, target_column="Цена"):
    """
    Подготавливает датасет для анализа временных рядов.
    :param file_path: Путь к файлу CSV
    :param target_column: Целевая колонка (по умолчанию "Цена")
    :return: DataFrame с преобразованными данными
    """
    # Загрузка данных
    data = pd.read_csv(file_path)

    # Удаление пробелов в названиях столбцов
    data.columns = data.columns.str.strip()

    # Преобразование столбца "Дата" в формат datetime
    data["Дата"] = pd.to_datetime(data["Дата"], format="%d.%m.%Y")

    # Установка "Дата" как индекса
    data.set_index("Дата", inplace=True)

    # Замена запятых на точки и преобразование числовых столбцов в float
    for col in data.columns:
        data[col] = (
            data[col].str.replace(",", ".").str.replace("K", "e3").str.replace("%", "")
        )
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Проверка на пропуски и удаление их (можно изменить логику обработки)
    data.dropna(inplace=True)
    data = data.drop(columns=["Изм. %"], axis=1)

    column_to_move = data.pop(target_column)
    data[target_column] = column_to_move

    # перевод в формат numpy
    data = data.values

    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=False)

    assert (
        len(train_data) > seq_length
    ), "Train data слишком короткий для заданного seq_length"
    assert (
        len(val_data) > seq_length
    ), "Validation data слишком короткий для заданного seq_length"
    assert (
        len(test_data) > seq_length
    ), "Test data слишком короткий для заданного seq_length"

    # Create datasets
    train_dataset = CryptoDataset(train_data, seq_length)
    val_dataset = CryptoDataset(val_data, seq_length)
    test_dataset = CryptoDataset(test_data, seq_length)

    input_dim = train_data.shape[1] - 1

    return train_dataset, val_dataset, test_dataset, input_dim  # , scaler


def download_mlflow_plot(run_name):

    """При условии завяски mlflow на корневые папки а не на S3.
    Скачивает графики"""

    # Создаем клиент
    client = MlflowClient()

    # Получаем эксперимент
    experiment = client.get_experiment_by_name(run_name)

    # Получаем последний запуск (RUN)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs:
        last_run_id = runs[0].info.run_id
        print(f"Последний Run ID: {last_run_id}")
    else:
        print("Нет запусков в эксперименте.")

    client = MlflowClient()
    local_dir = "./plots"
    artifact_path = "plots"
    client.download_artifacts(
        run_id=last_run_id, path=artifact_path, dst_path=local_dir
    )

    print(f"Артефакты из Run ID {last_run_id } сохранены в папке: {local_dir}")


@hydra.main(version_base="1.3.2", config_path="../configs", config_name="train")
def main(cfg: DictConfig):

    """Основная функция для запуска обучения"""

    print(f"Гиперпараметры из Hydra: {cfg}")

    # загрузка параметров с конфигов
    filepath = cfg.preprocess.filepath
    seq_length = cfg.preprocess.seq_length
    batch_size = cfg.preprocess.batch_size
    hidden_dim = cfg.model.hidden_dim
    output_dim = cfg.model.output_dim
    num_layers = cfg.model.num_layers
    lr = cfg.training.lr
    max_epochs = cfg.training.max_epochs

    # Инициализация MLflow
    run_name = "Crypto_Price_Prediction"
    mlflow.start_run(run_name=run_name)

    # Логируем гиперпараметры
    mlflow.log_params(
        {
            "filepath": cfg.preprocess.filepath,
            "seq_length": cfg.preprocess.seq_length,
            "batch_size": cfg.preprocess.batch_size,
            "hidden_dim": cfg.model.hidden_dim,
            "output_dim": cfg.model.output_dim,
            "num_layers": cfg.model.num_layers,
            "lr": cfg.training.lr,
            "max_epochs": cfg.training.max_epochs,
        }
    )

    # Prepare data
    train_dataset, val_dataset, _, input_dim = preprocess_time_series_data(
        filepath, seq_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = CryptoPricePredictor(input_dim, hidden_dim, output_dim, num_layers, lr)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # связываемся с логером
    mlflow_logger = MLFlowLogger(
        experiment_name="Crypto_Price_Prediction", tracking_uri="http://localhost:5050"
    )

    # Training
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=1,
        logger=mlflow_logger,
    )
    trainer.fit(model, train_loader, val_loader)

    # При условии отключения logger=mlflow_logger данные по модели будут сохраняться
    # через данный вызов
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath="checkpoints/",  # Папка для сохранения чекпоинтов
        filename="crypto-model-{epoch:02d}-{val_loss:.2f}",  # Формат имени файла
    )

    # download_mlflow_plot(run_name)


if __name__ == "__main__":
    main()
