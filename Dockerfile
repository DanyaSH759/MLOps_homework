# Базовый образ с Python
FROM python:3.10-slim

# Устанавливаем утилиты для работы
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем pyproject.toml и poetry.lock для установки зависимостей
COPY pyproject.toml poetry.lock /app/

# Устанавливаем зависимости
RUN poetry config virtualenvs.create false && poetry install --no-dev

# Копируем всё из текущей директории в контейнер
COPY . /app

# Устанавливаем переменные окружения для Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Команда по умолчанию (будет переопределяться при запуске контейнера)
CMD ["python"]
