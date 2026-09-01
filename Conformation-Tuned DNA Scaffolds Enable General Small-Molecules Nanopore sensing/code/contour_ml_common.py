from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


FEATURES = (
    "I/I0",
    "ΔSD",
    "Skew",
    "Median I/I0",
    "Dwell time (ms)",
    "Kurt",
    "IQR I/I0",
    "CV",
)
LABEL = "Analyte"
EIGHT_ANALYTE_CLASSES = ("Mal", "IsoMal", "Mal3", "IsoMal3", "3-DG", "5-AMF", "5-EMF", "5-HMF")
ESTROGEN_CLASSES = ("NIL", "EE2", "MES")

_LEGACY_COLUMN_NAMES = {
    "Analyte": ("English abbreviation",),
    "ΔSD": ("RMS (pA)",),
    "Dwell time (ms)": ("Dwell Time (ms)",),
    "Kurt": ("Kurtosis",),
}
_LEGACY_LABEL_VALUES = {
    "DP2": "Mal",
    "IDP2": "IsoMal",
    "DP3": "Mal3",
    "IDP3": "IsoMal3",
}


def load_feature_table(input_path: Path, class_order: Iterable[str], sheet: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load an external feature table without bundling or generating any data."""
    df = pd.read_excel(input_path, sheet_name=sheet)
    df = _normalise_column_names(df)

    required = [LABEL, *FEATURES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[LABEL] = df[LABEL].astype(str).replace(_LEGACY_LABEL_VALUES)
    class_order = tuple(class_order)
    df = df[df[LABEL].astype(str).isin(class_order)].copy()
    if df.empty:
        raise ValueError(f"No rows for the requested classes were found: {list(class_order)}")

    df[LABEL] = pd.Categorical(df[LABEL].astype(str), categories=class_order, ordered=True)
    sort_columns = [LABEL]
    if "Number" in df.columns:
        sort_columns.append("Number")
    df = df.sort_values(sort_columns).reset_index(drop=True)

    counts = df[LABEL].value_counts(sort=False)
    absent = [label for label in class_order if counts.get(label, 0) == 0]
    if absent:
        raise ValueError(f"Each requested class must have data. Missing classes: {absent}")

    X = df[list(FEATURES)].astype(float).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Feature columns contain non-finite values.")
    label_to_id = {label: index for index, label in enumerate(class_order)}
    y = df[LABEL].astype(str).map(label_to_id).to_numpy(dtype=int)
    return df, X, y


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for published_name, legacy_names in _LEGACY_COLUMN_NAMES.items():
        if published_name in df.columns:
            continue
        matches = [name for name in legacy_names if name in df.columns]
        if len(matches) == 1:
            df = df.rename(columns={matches[0]: published_name})
    return df


def build_eight_analyte_models(random_state: int = 42) -> dict[str, object]:
    """Return the 12 individual classifiers and the reported Stack classifier."""
    base = {
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "Logistic-EN": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.2,
                C=2.0,
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "QDA": make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.05)),
        "GaussianNB": GaussianNB(),
        "SVM-RBF": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=random_state),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=2,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=2,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, random_state=random_state),
            n_estimators=180,
            learning_rate=0.45,
            random_state=random_state,
        ),
        "HistGBDT": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.055,
            max_leaf_nodes=15,
            random_state=random_state,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=180,
            num_leaves=15,
            max_depth=4,
            learning_rate=0.055,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=2,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=160,
            depth=4,
            learning_rate=0.055,
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=2,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.055,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=2,
        ),
    }
    base["Stack"] = build_stack_model(random_state, base)
    return base


def build_stack_model(random_state: int = 42, base_models: dict[str, object] | None = None) -> StackingClassifier:
    base_models = base_models or {
        "SVM-RBF": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=random_state),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=2,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.055,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=2,
        ),
        "GaussianNB": GaussianNB(),
    }
    return StackingClassifier(
        estimators=[
            ("svm", base_models["SVM-RBF"]),
            ("rf", base_models["Random Forest"]),
            ("xgb", base_models["XGBoost"]),
            ("gnb", base_models["GaussianNB"]),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        stack_method="predict_proba",
        cv=3,
    )


def evaluate_holdout(
    X: np.ndarray,
    y: np.ndarray,
    models: dict[str, object],
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[dict[str, int | str]] = []
    for name, model in models.items():
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        pred = fitted.predict(X_validation)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_validation, pred),
                "macro_f1": f1_score(y_validation, pred, average="macro"),
                "macro_precision": precision_score(y_validation, pred, average="macro", zero_division=0),
                "macro_recall": recall_score(y_validation, pred, average="macro", zero_division=0),
                "n_train": int(len(y_train)),
                "n_validation": int(len(y_validation)),
            }
        )
        if name == "Stack":
            prediction_rows = [
                {"row_index": int(index), "true_class_id": int(true), "predicted_class_id": int(predicted)}
                for index, (true, predicted) in enumerate(zip(y_validation, pred))
            ]
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def evaluate_stack_oof(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[dict[str, int]]]:
    n_classes = int(np.max(y)) + 1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    predictions = np.zeros_like(y)
    probabilities = np.zeros((len(y), n_classes), dtype=float)
    folds: list[dict[str, int]] = []
    for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        model = clone(build_stack_model(random_state))
        model.fit(X[train_index], y[train_index])
        predictions[test_index] = model.predict(X[test_index])
        probabilities[test_index] = model.predict_proba(X[test_index])
        folds.append({"fold": fold, "train_n": int(len(train_index)), "test_n": int(len(test_index))})

    metrics = pd.DataFrame(
        [
            {
                "model": "Stack",
                "validation": "5-fold out-of-fold",
                "n_events": int(len(y)),
                "accuracy": accuracy_score(y, predictions),
                "macro_f1": f1_score(y, predictions, average="macro"),
                "macro_precision": precision_score(y, predictions, average="macro", zero_division=0),
                "macro_recall": recall_score(y, predictions, average="macro", zero_division=0),
                "log_loss": log_loss(y, probabilities, labels=np.arange(n_classes)),
            }
        ]
    )
    return metrics, predictions, probabilities, folds


def save_holdout_run(output_dir: Path, metrics: pd.DataFrame, predictions: pd.DataFrame, class_order: Iterable[str], random_state: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    if not predictions.empty:
        class_order = tuple(class_order)
        predictions = predictions.copy()
        predictions["true_class"] = predictions.pop("true_class_id").map(lambda value: class_order[int(value)])
        predictions["predicted_class"] = predictions.pop("predicted_class_id").map(lambda value: class_order[int(value)])
        predictions.to_csv(output_dir / "stack_validation_predictions.csv", index=False)
    best = metrics.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]["model"]
    write_manifest(
        output_dir,
        {
            "analysis": "eight-analyte holdout classification",
            "features": list(FEATURES),
            "class_order": list(class_order),
            "training_validation_split": "stratified 3:1",
            "random_state": random_state,
            "models": list(metrics["model"]),
            "best_model_by_macro_f1_then_accuracy": best,
            "stack_meta_learner": "LogisticRegression(max_iter=2000, class_weight='balanced')",
        },
    )


def save_oof_run(
    output_dir: Path,
    metrics: pd.DataFrame,
    y: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_order: Iterable[str],
    folds: list[dict[str, int]],
    random_state: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_order = tuple(class_order)
    metrics.to_csv(output_dir / "stack_model_metrics.csv", index=False)
    pd.DataFrame(
        confusion_matrix(y, predictions, labels=np.arange(len(class_order))),
        index=class_order,
        columns=class_order,
    ).to_csv(output_dir / "stack_confusion_matrix_counts.csv")
    prediction_table = pd.DataFrame(
        {
            "true_class": [class_order[int(value)] for value in y],
            "predicted_class": [class_order[int(value)] for value in predictions],
        }
    )
    for index, label in enumerate(class_order):
        prediction_table[f"prob_{label}"] = probabilities[:, index]
    prediction_table.to_csv(output_dir / "stack_oof_predictions.csv", index=False)
    write_manifest(
        output_dir,
        {
            "analysis": "three-analyte Stack classification",
            "features": list(FEATURES),
            "class_order": list(class_order),
            "outer_validation": "StratifiedKFold(n_splits=5, shuffle=True)",
            "random_state": random_state,
            "base_learners": ["SVM-RBF", "Random Forest", "XGBoost", "GaussianNB"],
            "stack_meta_learner": "LogisticRegression(max_iter=2000, class_weight='balanced')",
            "stack_method": "predict_proba",
            "inner_cv": 3,
            "folds": folds,
        },
    )


def write_manifest(output_dir: Path, payload: dict[str, object]) -> None:
    (output_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

