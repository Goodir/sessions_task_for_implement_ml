import numpy as np
import onnxruntime as ort
import pandas as pd
import requests

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "model" / "UCI_Credit_Card.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "model_v1.onnx"


print('!'*100)
print('Проверка доступности сервера:')
server_status = 0
try:
    response_get = requests.get('http://127.0.0.1:8080')

    print('Код ответа по get запросу страницы "/":', response_get.status_code)
    if response_get.status_code == 200:
        print('Текст ответа:', response_get.text)
    elif response_get.status_code == 404:
        print('Контейнер запущен, но route не найден')
    elif response_get.status_code == 500:
        print('Контейнер запущен, но внутри ошибка приложения')
    print('*'*100)
    print('Проверка "/health"')

    response_get = requests.get('http://127.0.0.1:8080/health')

    print('Код ответа по get запросу страницы "health":', response_get.status_code)
    if response_get.status_code == 200:
        server_status = 1
        print('Json ответа:', response_get.json())
    elif response_get.status_code == 404:
        print('Контейнер запущен, но route не найден')
    elif response_get.status_code == 500:
        print('Контейнер запущен, но внутри ошибка приложения')

except requests.exceptions.ConnectionError:
    print('API недоступен. Возможно, контейнер не запущен или порт указан неверно')

except requests.exceptions.Timeout:
    print('API не ответил за отведенное время')

except requests.exceptions.RequestException as e:
    print('Ошибка при запросе к API:', e)

print('-'*100)
print('Проверка доступности данных:')
data_check = 0

try:
    test_data = pd.read_csv(DATA_PATH).sample(1, random_state=42).drop(['default.payment.next.month'], axis=1)
    onnx_local_model = ort.InferenceSession(MODEL_PATH)
    json_data_for_onnx = {"features": test_data.to_numpy(dtype="float32").tolist()}
    data_check = 1
    print('Файлы с моделью и данными обнаружены')
except FileNotFoundError:
    print('Файл с моделью и/или данными не обнаружен')

print('-'*100)
if not server_status:
    print('Перед проверкой к post запросу модели следует разобраться с сервером')
elif not data_check:
    print('Перед проверкой к post запросу модели следует разобраться с доступностью файлов')
else:
    print('Проверка post запроса к модели, и самой модели:')
    
    respons_post = requests.post('http://127.0.0.1:8080/predict', json=json_data_for_onnx)
    print('Код ответа по post запросу страницы "predict":', respons_post.status_code)
    if respons_post.status_code == 200:
        print('Json ответа:', respons_post.json())
        post_status=1
    elif respons_post.status_code == 404:
        print('Контейнер запущен, но route не найден')
    elif respons_post.status_code == 500:
        print('Контейнер запущен, но внутри ошибка приложения')
print('!'*100)