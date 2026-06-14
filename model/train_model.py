import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

import onnxruntime as ort
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "model" / "UCI_Credit_Card.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "model_v1.onnx"
rs = 42

df = pd.read_csv(DATA_PATH)
num_standars_col = ['LIMIT_BAL', *[f'PAY_AMT{i}' for i in [1, 2, 3, 4, 5, 6]]]
num_pass_cols = [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]

X_train, X_test, y_train, y_test = train_test_split(df.drop(['default.payment.next.month'], axis=1), df['default.payment.next.month'], test_size=0.2, random_state=rs)
feature_order = list(X_train.columns)

num_standars_col_ix = [feature_order.index(col) for col in num_standars_col]
num_pass_cols_ix = [feature_order.index(col) for col in num_pass_cols]

num_scale_pipe = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num_scaler', num_scale_pipe, num_standars_col_ix),
    ('numm_pass', 'passthrough', num_pass_cols_ix),
])

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=rs))
])

param_grid = {
    "classifier__n_estimators": [200],
    "classifier__max_depth": [10],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
    "classifier__max_features": ["sqrt"],
    "classifier__bootstrap": [True]
}

print('Обучение grid поиска')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

best_pipeline = grid.best_estimator_
print()
print('_'*100)
print()
print('f1 score на тесте, у модели:', f1_score(y_test, best_pipeline.predict(X_test), average='macro'))


print('Загрузка модели в onnx формат')
onnx_model = to_onnx(
    best_pipeline,
    initial_types=[
        ("features", FloatTensorType([None, len(feature_order)]))
    ],
    target_opset=17)


print()
print('Проверка что загруженная onnx модель, дает такой результат как только что обученная | Проверка на тех же тестовых данных')

with open(MODEL_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

session = ort.InferenceSession(MODEL_PATH)
inputs = X_test.to_numpy(dtype='float32')

outputs = session.run(None, {'features':inputs})

pred_sklearn = grid.best_estimator_.predict(X_test)
pred_onnx = outputs[0]

print('Средняя ошибка onnx vs sklearn:', (pred_sklearn != pred_onnx).mean())

print('f1 score у onnx на тесте:', f1_score(y_test, outputs[0], average='macro'))

diff_mask = pred_sklearn != pred_onnx

print('Количество ошибок относительно sklearn модели, и общее количество объектов:', diff_mask.sum(), len(pred_sklearn))