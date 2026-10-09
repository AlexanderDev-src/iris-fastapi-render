import importlib.util
import json

import numpy as np
from fastapi.testclient import TestClient
from joblib import load

from model import PROJECT, X_test, feature_keys, iris, model_path

loaded_model = load(model_path)

spec = importlib.util.spec_from_file_location("iris_api", PROJECT / "main.py")
api_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_module)
local_client = TestClient(api_module.app)

sample = dict(zip(feature_keys, [5.1, 3.5, 1.4, 0.2]))
assert local_client.get("/health").status_code == 200
assert local_client.get("/docs").status_code == 200
response = local_client.post("/predict", json=sample)
assert response.status_code == 200
result = response.json()
assert result["predicted_class_index"] == int(
    loaded_model.predict([list(sample.values())])[0]
)
print(json.dumps(result, indent=2, ensure_ascii=False))

bad_payloads = [
    {**sample, "sepal_length": -1},
    {**sample, "petal_width": 0},
    {**sample, "petal_length": "hello"},
    {**sample, "sepal_width": None},
    {**sample, "sepal_length": "NaN"},
    {**sample, "sepal_length": "Infinity"},
    {key: value for key, value in sample.items() if key != "petal_width"},
    {**sample, "unknown_field": 1},
]
for bad in bad_payloads:
    r = local_client.post("/predict", json=bad)
    assert r.status_code == 422, r.text
print(f"ผ่าน validation tests: {len(bad_payloads)} กรณี")

for row in X_test:
    remote = local_client.post(
        "/predict", json=dict(zip(feature_keys, row.tolist()))
    ).json()
    assert remote["predicted_class_index"] == int(loaded_model.predict([row])[0])
    expected = loaded_model.predict_proba([row])[0]
    actual = [
        remote["probabilities"][str(iris.target_names[int(k)])]
        for k in loaded_model.classes_
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
print(f"ผล API ตรงกับโมเดลทั้ง {len(X_test)} ตัวอย่างใน test set")
