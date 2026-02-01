"""
Preprocessing utilites
- Import precleaned data
- Impute NULL Values
- Scale Data
- Encode Categorical data
- Train Test Split
"""
# imports
import pandas as pd
import numpy as np
import pathlib as Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.config import DATA_DIR, CLEANED_DATA_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COL
from typing import List, Tuple

# Load pre-cleaned data
def load_data(data_path: Path = CLEANED_DATA_PATH):
    
    """
    This will load pre-cleaned data
    Remove leading or lagging hite spaces from column name
    Return Clean data
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()
    """
    df = pd.read_csv(data_path)
   
   # Normalize columns
    df.columns = [str(c).strip() for c in df.columns.to_list()]
    
# Extract categorical and numerical columns
def _extract_cat_cols_num_cols(df:pd.DataFrame):
    cat_cols = df.select_dtypes(include=["object", "category","bool"]).columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()
        
    return cat_cols, num_cols

# Build preprocessor
def build_preprocessor(df_or_x: pd.DataFrame):
    # Create back up
    df = df_or_x.copy()
    
    # If target column is present drop the column
    if TARGET_COL in df.columns:
        df = df.drop(columns= TARGET_COL)
    
    else:
        df = df
           
    # Extract categorical and numerical columns    
    cat_cols, num_cols = _extract_cat_cols_num_cols(df)
    
    # Numeric Pipeline: Impute -> Scale
    num_transformer = Pipeline(steps =[
        
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical Pipeline: Imputing -> OneHotEncoding
    cat_transformer = Pipeline(steps= [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine the pipelines 
    transformers = []
    if cat_cols:
        transformers.append(("cat",cat_transformer,cat_cols))
    
    if num_cols:
        transformers.append(("num",num_transformer,num_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    
    return preprocessor

# Train Test Split
def split_data(df: pd.DataFrame):
    df = df.copy()
    # If Target Column is missing
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target Column {TARGET_COL} is not found in DataFrame columns: {df.columns.to_list()}")
    
    X = df.drop(columns= TARGET_COL)
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,TEST_SIZE= TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train, X_test, y_train, y_test

print("Preprocessing is Executed")