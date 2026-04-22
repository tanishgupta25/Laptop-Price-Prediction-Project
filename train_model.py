"""
train_model.py
--------------
Full ML pipeline:
  1. Load & preprocess data
  2. Feature engineering
  3. Train LinearRegression, RandomForest, GradientBoosting
  4. Evaluate MAE, RMSE, R²
  5. RandomizedSearchCV hyperparameter tuning on best model
  6. Save best pipeline as model.pkl
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  LAPTOP PRICE PREDICTION — MODEL TRAINING")
print("=" * 60)

df = pd.read_csv("laptop_augmented.csv", encoding="utf-8")
print(f"\n✔ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("\n[1/5] Feature Engineering ...")


def parse_ram(ram_str):
    try:
        return int(str(ram_str).replace("GB", "").strip())
    except Exception:
        return 8


def parse_weight(w_str):
    try:
        return float(str(w_str).replace("kg", "").strip())
    except Exception:
        return 2.0


def extract_cpu_brand(cpu_str):
    cpu_str = str(cpu_str)
    if "Intel" in cpu_str:
        return "Intel"
    elif "AMD" in cpu_str:
        return "AMD"
    return "Other"


def extract_cpu_tier(cpu_str):
    cpu_str = str(cpu_str)
    if "i9" in cpu_str or "i9" in cpu_str:
        return "High"
    elif "i7" in cpu_str or "Ryzen 7" in cpu_str:
        return "High"
    elif "i5" in cpu_str or "Ryzen 5" in cpu_str:
        return "Mid"
    elif "i3" in cpu_str or "Ryzen 3" in cpu_str:
        return "Entry"
    return "Entry"


def extract_cpu_ghz(cpu_str):
    try:
        tokens = str(cpu_str).split()
        for t in tokens:
            if "GHz" in t:
                return float(t.replace("GHz", ""))
    except Exception:
        pass
    return 2.0


def extract_ssd(mem_str):
    ssd = 0
    parts = str(mem_str).upper().split("+")
    for p in parts:
        nums = [int(x) for x in p.split() if x.isdigit()]
        if nums and ("SSD" in p or "FLASH" in p or "NVME" in p):
            size = nums[0]
            if "TB" in p:
                size *= 1024
            ssd += size
    return ssd


def extract_hdd(mem_str):
    hdd = 0
    parts = str(mem_str).upper().split("+")
    for p in parts:
        nums = [int(x) for x in p.split() if x.isdigit()]
        if nums and "HDD" in p:
            size = nums[0]
            if "TB" in p:
                size *= 1024
            hdd += size
    return hdd


def extract_gpu_tier(gpu_str):
    gpu_str = str(gpu_str)
    if "RTX" in gpu_str or "Quadro" in gpu_str:
        return "High"
    elif any(x in gpu_str for x in ["GTX 1060", "GTX 1070", "GTX 1080", "RX 580"]):
        return "Mid-High"
    elif any(x in gpu_str for x in ["GTX 1050", "940MX", "Radeon"]):
        return "Mid"
    return "Integrated"


def extract_resolution_category(res_str):
    res_str = str(res_str)
    if "4K" in res_str or "3840" in res_str:
        return "4K"
    elif any(x in res_str for x in ["2560", "2880", "2304", "2960"]):
        return "QHD"
    elif "1920" in res_str or "Full HD" in res_str:
        return "FHD"
    return "HD"


df["RAM_GB"] = df["Ram"].apply(parse_ram)
df["Weight_kg"] = df["Weight"].apply(parse_weight)
df["CPU_Brand"] = df["Cpu"].apply(extract_cpu_brand)
df["CPU_Tier"] = df["Cpu"].apply(extract_cpu_tier)
df["CPU_GHz"] = df["Cpu"].apply(extract_cpu_ghz)
df["SSD_GB"] = df["Memory"].apply(extract_ssd)
df["HDD_GB"] = df["Memory"].apply(extract_hdd)
df["GPU_Tier"] = df["Gpu"].apply(extract_gpu_tier)
df["Resolution_Cat"] = df["ScreenResolution"].apply(extract_resolution_category)

print("   ✔ Feature engineering complete")

# ── 3. Define Features & Target ───────────────────────────────────────────────
CATEGORICAL = ["Company", "TypeName", "CPU_Brand", "CPU_Tier",
               "GPU_Tier", "Resolution_Cat", "OpSys"]
NUMERIC = ["Inches", "RAM_GB", "Weight_kg", "CPU_GHz", "SSD_GB", "HDD_GB"]
TARGET = "Price_euros"

df = df.dropna(subset=CATEGORICAL + NUMERIC + [TARGET])
X = df[CATEGORICAL + NUMERIC]
y = np.log1p(df[TARGET])   # log-transform target for better regression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[2/5] Data Split: train={len(X_train)}, test={len(X_test)}")

# ── 4. Preprocessing Pipeline ─────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])


def make_pipeline(estimator):
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def evaluate(pipe, X_t, y_t, name):
    pred_log = pipe.predict(X_t)
    pred = np.expm1(pred_log)
    actual = np.expm1(y_t)
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    print(f"   {name:<30}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  R²={r2:.4f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "pipeline": pipe}


# ── 5. Train & Compare Models ─────────────────────────────────────────────────
print("\n[3/5] Training & Evaluating Models ...")
print("   " + "-" * 60)

results = []

# Linear Regression
lr_pipe = make_pipeline(LinearRegression())
lr_pipe.fit(X_train, y_train)
results.append(evaluate(lr_pipe, X_test, y_test, "Linear Regression"))

# Random Forest
rf_pipe = make_pipeline(
    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
)
rf_pipe.fit(X_train, y_train)
results.append(evaluate(rf_pipe, X_test, y_test, "Random Forest"))

# Gradient Boosting
gb_pipe = make_pipeline(
    GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                               max_depth=5, random_state=42)
)
gb_pipe.fit(X_train, y_train)
results.append(evaluate(gb_pipe, X_test, y_test, "Gradient Boosting"))

# ── 6. Select Best Model ──────────────────────────────────────────────────────
best_result = max(results, key=lambda r: r["r2"])
print(f"\n   ✔ Best model: {best_result['name']}  (R² = {best_result['r2']:.4f})")

# ── 7. Hyperparameter Tuning ──────────────────────────────────────────────────
print("\n[4/5] Hyperparameter Tuning (RandomizedSearchCV) ...")

if "Random Forest" in best_result["name"]:
    param_dist = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    }
    base_estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
else:
    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.05, 0.1, 0.15],
        "model__max_depth": [3, 4, 5, 6],
        "model__min_samples_split": [2, 5],
        "model__subsample": [0.8, 0.9, 1.0],
    }
    base_estimator = GradientBoostingRegressor(random_state=42)

tuning_pipe = make_pipeline(base_estimator)
search = RandomizedSearchCV(
    tuning_pipe,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)

print(f"\n   Best CV R²: {search.best_score_:.4f}")
print(f"   Best params: {search.best_params_}")

tuned_result = evaluate(search.best_estimator_, X_test, y_test, "Tuned Best Model")

# ── 8. Final Model Selection ──────────────────────────────────────────────────
final_pipeline = (
    search.best_estimator_
    if tuned_result["r2"] > best_result["r2"]
    else best_result["pipeline"]
)
final_r2 = max(tuned_result["r2"], best_result["r2"])
print(f"\n   ✔ Final Model R²: {final_r2:.4f}")

# ── 9. Save Artifacts ─────────────────────────────────────────────────────────
print("\n[5/5] Saving Artifacts ...")

joblib.dump(final_pipeline, "model.pkl")
print("   ✔ model.pkl saved")

# Save metadata for app.py
meta = {
    "categorical_features": CATEGORICAL,
    "numeric_features": NUMERIC,
    "final_r2": round(final_r2, 4),
    "companies": sorted(df["Company"].unique().tolist()),
    "types": sorted(df["TypeName"].unique().tolist()),
    "os_options": sorted(df["OpSys"].unique().tolist()),
    "ram_options": sorted(df["RAM_GB"].unique().tolist()),
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("   ✔ model_meta.json saved")

print("\n" + "=" * 60)
print(f"  Training complete!  Final R² = {final_r2:.4f}")
print("=" * 60)
