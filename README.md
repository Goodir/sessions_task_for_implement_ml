# Default prediction ML service

Учебный MLOps-проект по внедрению модели машинного обучения в production-like окружение

Сервис предсказывает вероятность дефолта клиента по данным кредитной карты. Модель обучена на датасете `Default of Credit Card Clients` и упакована в Flask API. Для инференса используется ONNX-модель, сервис запускается в Docker-контейнере через Docker Compose

## Что есть в проекте

* обучение модели и сохранение в ONNX
* Flask API с endpoint'ами `/health` и `/predict`
* A/B-интерфейс через поле `model_version`
* запуск через Docker и Docker Compose
* uWSGI + NGINX через образ `tiangolo/uwsgi-nginx-flask`
* JSON-логирование запросов к `/predict`
* тестовый скрипт для проверки API
* описание MLOps-концептов и A/B-тестирования

## Стек

```text
Python
Flask
scikit-learn
ONNX Runtime
Docker
Docker Compose
uWSGI
NGINX
```

## Структура проекта

```text
.
├── app
│   ├── api.py
│   └── model_handler.py
├── model
│   ├── train_model.py
│   ├── model_v1.onnx
│   ├── model_v2.onnx
│   └── UCI_Credit_Card.csv
├── tests
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── uwsgi.ini
├── MLops_CONCEPT.md
├── README.md
└── README_AB.md
```

## Проверенные окружения

Проект проверялся на двух окружениях:

```text
macOS, Python 3.12
Fedora Server в UTM, Python 3.14
```

В обоих случаях сервис запускается, контейнер собирается, API отвечает и тестовый скрипт проходит

## Быстрый запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/Goodir/sessions_task_for_implement_ml.git
cd sessions_task_for_implement_ml
```

### 2. Создать виртуальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для Windows:

```bash
.venv\Scripts\activate
```

## Запуск через Docker Compose

Основной вариант запуска проекта:

```bash
docker compose up --build
```

После запуска API будет доступен по адресу:

```text
http://127.0.0.1:8080
```

Проверка health endpoint:

```bash
curl http://127.0.0.1:8080/health
```

Ожидаемый ответ:

```json
{"status":"healthy"}
```

## Проверка API

Когда контейнер уже запущен, можно проверить сервис тестовым скриптом:

```bash
python tests/test_api.py
```

Скрипт проверяет:

* доступность сервера
* endpoint `/health`
* наличие данных и модели
* POST-запрос к `/predict`

## Пример запроса к `/predict`

Запрос лучше запускать в одну строку, чтобы не было проблем с переносами в терминале

```bash
curl -X POST "http://127.0.0.1:8080/predict" -H "Content-Type: application/json" --data-raw '{"model_version": "v1", "features": [[1, 20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2, 3913, 3102, 689, 0, 0, 0, 0, 689, 0, 0, 0, 0]]}'
```

Пример ответа:

```json
{
  "model_version": "v1",
  "prediction": [1],
  "probability": [
    {
      "0": 0.33265841007232666,
      "1": 0.6673415899276733
    }
  ],
  "request_id": "2966c45c-ffc0-406b-b170-8465618560ad"
}
```

`prediction = 0` означает, что модель не предсказывает дефолт
`prediction = 1` означает, что модель предсказывает дефолт

`probability` показывает вероятность классов `0` и `1`
`model_version` показывает версию модели, которая использовалась для предсказания
`request_id` нужен, чтобы связать ответ API с записью в логах

Сейчас в проекте используется одна модель `v1`. Поле `model_version` добавлено как интерфейсная заготовка под A/B-тестирование

## Повторное обучение модели

В репозитории уже есть сохраненная ONNX-модель, поэтому для обычного запуска переобучать ее не нужно

Если нужно полностью повторить обучение и заново сохранить модель:

```bash
python model/train_model.py
```

Скрипт обучает sklearn pipeline, сохраняет модель в `model/model_v1.onnx`, а потом проверяет, что ONNX-модель дает такой же результат, как исходная sklearn-модель

После переобучения лучше пересобрать контейнер:

```bash
docker compose up --build
```

И снова проверить API:

```bash
python tests/test_api.py
```

## Логи

В проекте добавлено JSON-логирование запросов к `/predict`

Посмотреть логи контейнера:

```bash
docker compose logs ml-api
```

Если нужны только структурированные логи приложения:

```bash
docker compose logs ml-api | grep APP_LOG
```

В лог пишутся:

```text
timestamp
request_id
endpoint
status_code
model_version
input_rows
prediction
duration_ms
```

Полные признаки клиента в лог не пишутся, потому что для кредитного скоринга это могут быть чувствительные данные


## A/B-интерфейс

API поддерживает работу с версиями модели через поле `model_version` 

Полную документацию по A/B тестированию можно найти в `README_AB.md`

Пример явного выбора модели:

```bash
curl -X POST "http://127.0.0.1:8080/predict" -H "Content-Type: application/json" --data-raw '{"model_version": "v1", "features": [[1, 20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2, 3913, 3102, 689, 0, 0, 0, 0, 689, 0, 0, 0, 0]]}'
```

Также можно не передавать `model_version`, а передать `client_id`. Тогда сервис сам распределит клиента между `v1` и `v2` через hash:

```bash
curl -X POST "http://127.0.0.1:8080/predict" -H "Content-Type: application/json" --data-raw '{"client_id": "client_001", "features": [[1, 20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2, 3913, 3102, 689, 0, 0, 0, 0, 689, 0, 0, 0, 0]]}'
```

В ответе API всегда возвращает `model_version`, чтобы было понятно, какая версия модели использовалась

## Остановка сервиса

```bash
docker compose down
```
