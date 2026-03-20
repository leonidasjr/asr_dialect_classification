import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# -----------------------------
# Configuration
# -----------------------------
RANDOM_SEED = 1
TEST_SIZE = 0.20

INPUT_FILE = "data/asr_dataset.csv"
REPORT_FILE = "outputs/classification_report.txt"
CM_FILE = "outputs/confusion_matrices.csv"
FI_FILE = "outputs/feature_importance.csv"
SPLIT_FILE = "data/train_test_split_ids.csv"

FEATURES = [
    "f0sd",
    "f0SAQ",
    "df0mean_pos",
    "df0sd_pos",
    "sl_LTAS_alpha",
    "cvint",
    "pause_sd",
    "pause_meandur",
    "pause_rate"
]

# -----------------------------
# Read data
# -----------------------------
df = pd.read_csv(INPUT_FILE)

required_cols = ["sample_id", "dialect"] + FEATURES
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Encode labels explicitly
label_map = {"PB": 0, "SP": 1}
df["dialect_num"] = df["dialect"].map(label_map)

X = df[FEATURES].copy()
y = df["dialect_num"].copy()

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X_scaled,
    y,
    df["sample_id"],
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

# Save exact split
split_df = pd.concat([
    pd.DataFrame({"sample_id": id_train.values, "dialect": df.set_index("sample_id").loc[id_train, "dialect"].values, "set": "train"}),
    pd.DataFrame({"sample_id": id_test.values, "dialect": df.set_index("sample_id").loc[id_test, "dialect"].values, "set": "test"})
], ignore_index=True)
os.makedirs("data", exist_ok=True)
split_df.to_csv(SPLIT_FILE, index=False)

# -----------------------------
# Models and grids
# -----------------------------
model_grid_list = [
    ("LDA", LinearDiscriminantAnalysis(), {
        "solver": ["lsqr", "eigen"],
        "shrinkage": [None, "auto"]
    }),
    ("kNN", KNeighborsClassifier(), {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    }),
    ("DT", DecisionTreeClassifier(random_state=RANDOM_SEED), {
        "max_depth": [None, 5, 10],
        "criterion": ["gini", "entropy"]
    }),
    ("RF", RandomForestClassifier(random_state=RANDOM_SEED), {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "max_features": ["sqrt"]
    }),
    ("SVM", SVC(random_state=RANDOM_SEED), {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }),
    ("GBM", GradientBoostingClassifier(random_state=RANDOM_SEED), {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0]
    })
]

best_models = []
best_params = {}

for name, model, grid_params in model_grid_list:
    grid = GridSearchCV(
        estimator=model,
        param_grid=grid_params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_models.append((name, grid.best_estimator_))
    best_params[name] = grid.best_params_

# -----------------------------
# Evaluation
# -----------------------------
os.makedirs("outputs", exist_ok=True)

cm_rows = []
report_lines = []

for name, model in best_models:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    err = 1 - acc
    cm = confusion_matrix(y_test, preds, labels=[0, 1])

    report_lines.append(f"===== {name} =====")
    report_lines.append(f"Best parameters: {best_params[name]}")
    report_lines.append(f"Accuracy: {acc:.4f}")
    report_lines.append(f"Error rate: {err:.4f}")
    report_lines.append("Confusion matrix [[PB, SP],[PB, SP]]:")
    report_lines.append(str(cm))
    report_lines.append(classification_report(y_test, preds, target_names=["PB", "SP"]))
    report_lines.append("")

    cm_rows.append({
        "model": name,
        "PB_pred_PB": int(cm[0, 0]),
        "PB_pred_SP": int(cm[0, 1]),
        "SP_pred_PB": int(cm[1, 0]),
        "SP_pred_SP": int(cm[1, 1]),
        "accuracy": round(acc, 4),
        "error_rate": round(err, 4)
    })

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

pd.DataFrame(cm_rows).to_csv(CM_FILE, index=False)

# -----------------------------
# Feature importance from RF
# -----------------------------
rf_model = dict(best_models)["RF"]
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_

fi_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": importances
}).sort_values("importance", ascending=False)

fi_df.to_csv(FI_FILE, index=False)

# -----------------------------
# Save minimal environment metadata
# -----------------------------
meta = {
    "random_seed": RANDOM_SEED,
    "test_size": TEST_SIZE,
    "features": FEATURES,
    "best_parameters": best_params
}

with open("outputs/run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Classification completed.")
print(f"Report: {REPORT_FILE}")
print(f"Confusion matrices: {CM_FILE}")
print(f"Feature importance: {FI_FILE}")
print(f"Exact split: {SPLIT_FILE}")