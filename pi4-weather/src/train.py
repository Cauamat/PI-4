# src/train.py
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .preprocess import load_silver, make_features

def eval_thresholds(y_true, y_score, thresholds):
    rows = []
    for th in thresholds:
        y_pred = (y_score >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append({
            "threshold": float(th),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })
    return rows

def f_beta(p, r, beta=1.5):
    if (p + r) == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * (p * r) / (b2 * p + r)

def main():
    # ===== 1) Carregar e preparar features =====
    df = load_silver()
    df, feats, target = make_features(df)
    X, y = df[feats], df[target]

    # Split simples estratificado (se quiser, depois troque por split temporal)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # ===== 2) Modelo (com scaler e balanceamento) =====
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"  # "saga" também é ok
        ))
    ])

    clf.fit(Xtr, ytr)

    # ===== 3) Métricas padrão =====
    yhat = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    print(classification_report(yte, yhat, digits=3))
    print("ROC-AUC:", roc_auc_score(yte, proba))

    # ===== 4) Varredura de thresholds =====
    thresholds = np.round(np.linspace(0.10, 0.60, 6), 2)
    rows = eval_thresholds(yte, proba, thresholds)
    print("\n=== Threshold sweep ===")
    for r in rows:
        print(r)

    # Escolha automática priorizando recall (F1.5); ajuste se quiser
    best = max(rows, key=lambda r: f_beta(r["precision"], r["recall"], beta=1.5))
    print("\nEscolhido (F1.5):", best)

    # ===== 5) Salvar modelo + features + threshold =====
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": clf, "features": feats, "threshold": best["threshold"]},
        "models/rain_classifier.pkl"
    )
    print(f"Modelo salvo em models/rain_classifier.pkl com limiar={best['threshold']:.2f}")

if __name__ == "__main__":
    main()
