"""
Basic configuration and Path
"""

from pathlib import Path

## Root Directory
BASE_DIR:Path = Path(__file__).resolve().parents[1]

# Data and Model path
DATA_DIR:Path = Path(BASE_DIR)/"data"
DATA_PATH:Path = Path(DATA_DIR)/"Telco_Customer_Churn.csv"
CLEANED_DATA_PATH:Path = Path (DATA_DIR) / "cleaned_data.csv"

MODEL_DIR :Path = Path (BASE_DIR)/"models"
BEST_MODEL_PATH:Path = Path (MODEL_DIR) / "Best_model.joblib"
FEATURES_PATH:Path = Path (MODEL_DIR) / "Feature_columns.jason"

TARGET_COL:str = "Churn"

# Train ,Test and Cross validation
TEST_SIZE:float = 0.2
RANDOM_STATE:int = 42
CV_FOLD:int =5
N_JOBS: int = -1
SCORING:str = "roc_auc"

print("Configuration are successfully applied")