import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

df = pd.read_csv("hfp/heart_failure_clinical_records_dataset.csv")

features = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
]

X = df[features]
y = df["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)

y_prob = model.predict_proba(X_test_sc)[:, 1]


precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_idx_f1 = np.argmax(f1_scores)
threshold_f1 = pr_thresholds[best_idx_f1] if best_idx_f1 < len(pr_thresholds) else 0.5

print(f"Optimal F1 threshold: {threshold_f1:.4f}")
print(f"Best F1 score:        {f1_scores[best_idx_f1]:.4f}")

fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
best_idx_roc = np.argmin(np.abs(roc_thresholds - threshold_f1))


fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.scatter(
    fpr[best_idx_roc],
    tpr[best_idx_roc],
    color="green",
    zorder=5,
    s=100,
    label=f"F1 threshold = {threshold_f1:.3f}",
)
ax.set_xlabel("FP Rate")
ax.set_ylabel("TP Rate")
ax.set_title("Krzywa ROC")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()


sigmoid_features = features

for feat in sigmoid_features:
    feat_idx = features.index(feat)

    x_range = np.linspace(X[feat].min(), X[feat].max(), 300)

    X_sweep_raw = np.tile(scaler.mean_, (300, 1))
    X_sweep_raw[:, feat_idx] = x_range

    X_sweep_sc = scaler.transform(X_sweep_raw)
    y_sweep = model.predict_proba(X_sweep_sc)[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    feat_vals = X_test[feat].values
    ax.scatter(feat_vals, y_test.values, alpha=0.3, color="gray", zorder=2, label="Actual (0/1)")
    ax.plot(x_range, y_sweep, color="steelblue", lw=2.5, zorder=3, label="P(death)")
    ax.axhline(
        threshold_f1,
        color="green",
        linestyle="--",
        lw=1.5,
        label=f"F1 threshold = {threshold_f1:.3f}",
    )
    ax.set_xlabel(feat.replace("_", " ").title())
    ax.set_ylabel("P(DEATH_EVENT = 1)")
    ax.set_title(f"Sigmoid — {feat.replace('_', ' ').title()}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"sigmoid_{feat}.png", dpi=150)
    plt.show()


y_pred = (y_prob >= threshold_f1).astype(int)

print(f"\nClassification Report [threshold = {threshold_f1:.3f}]")
print(classification_report(y_test, y_pred, target_names=["Survived", "Died"]))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["Survived", "Died"]).plot(cmap="Blues", ax=ax)
ax.set_title(f"Confusion Matrix — F1 Threshold ({threshold_f1:.3f})")
plt.tight_layout()
plt.savefig("confusion_matrix_f1.png", dpi=150)
plt.show()
