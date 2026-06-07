import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import argparse
import os

# ============================================================
# DAGSHUB CONFIGURATION
# ============================================================
DAGSHUB_USERNAME = "Temlearnt"
DAGSHUB_REPO = "Eksperimen_SML_IPutuSuthaSatyawan"
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN", "")

print(f"Tracking URI: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
print(f"Token loaded: {len(DAGSHUB_TOKEN) > 0}")

if DAGSHUB_TOKEN:
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=200)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--min_samples_split", type=int, default=2)
args = parser.parse_args()

# Load dataset
df = pd.read_csv('dataset_preprocessing/Retail_Transactions_preprocessing.csv')
print(f"📊 Dataset shape: {df.shape}")

# ============= CLEAN DATA: Drop non-numeric columns =============
TARGET = 'Total_Cost'

# Drop kolom yang tidak diperlukan (non-numeric)
cols_to_drop = ['Product', 'Transaction_ID', 'Customer_Name', 'Date']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Encode categorical columns if any
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"✅ Encoded: {col}")

# Verify target exists
if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' not found in dataset. Columns: {df.columns.tolist()}")

X = df.drop(columns=[TARGET])
y = df[TARGET]
# ============================================================

print(f"📊 Features shape: {X.shape}")
print(f"📊 Features types:\n{X.dtypes.value_counts()}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("Workflow-CI-Experiment")

with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    mlflow.sklearn.log_model(model, "model")
    
    print(f"\n✅ Model trained!")
    print(f"   MAE: {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R2: {r2:.4f}")
    print(f"   Run ID: {run.info.run_id}")