import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from joblib import dump, load
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parent
model_path = PROJECT / "iris_random_forest.joblib"
feature_keys = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

if __name__ == "__main__":
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    dump(model, model_path)
    metadata = {
        "model_version": "iris-rf-v1",
        "feature_order": feature_keys,
        "target_names": iris.target_names.tolist(),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "random_state": 42,
        "python_version": platform.python_version(),
        "sklearn_version": version("scikit-learn"),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (PROJECT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    loaded_model = load(model_path)
    np.testing.assert_array_equal(model.predict(X_test), loaded_model.predict(X_test))
    print("โหลดกลับแล้วได้ผลตรงกับโมเดลเดิม")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
