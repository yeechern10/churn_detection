from pathlib import Path

MLFLOW_TRACKING_URI = "https://dagshub.com/yeeislazy/churn_detection.mlflow/"

EXPERIMENT_NAME = "churn_detection"
MODEL_NAME = "customer-churn-model"

# dirs
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# files
SCHEMA_FILE = ARTIFACTS_DIR / "schema.json"