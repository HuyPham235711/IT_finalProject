import pandas as pd
from sqlalchemy import create_engine
import os
import numpy as np

def load_table_from_postgres(table_name: str, schema: str, conn_env: str = "PG_CONN_STR"):
    """
    Load bảng từ PostgreSQL, convert numeric -> float32, fill NaN.
    Không scale lại vì data đã min-max chuẩn hóa sẵn.
    """
    conn_str = os.getenv(conn_env)
    if not conn_str:
        raise EnvironmentError(f"Missing environment variable {conn_env}")

    engine = create_engine(conn_str)
    query = f'SELECT * FROM "{schema}"."{table_name}" ORDER BY datetime ASC'
    df = pd.read_sql(query, engine)

    print(f"[INFO] Loaded {table_name} shape={df.shape}")
    print(df.isna().sum())

    # Chuyển kiểu dữ liệu
    df["datetime"] = pd.to_datetime(df["datetime"])
    numeric_cols = df.select_dtypes(include="number").columns

    # Fill NaN cho các cột kỹ thuật như sma14, rsi14
    df[numeric_cols] = df[numeric_cols].fillna(method="ffill").fillna(method="bfill")
    df[numeric_cols] = df[numeric_cols].astype(np.float32)

    return df


def make_time_windows(df: pd.DataFrame, features: list, target: str, lookback: int, stride: int = 1):
    """
    Tạo tensor 3D (samples, lookback, features) và target 1D cho LSTM.
    """
    X, y = [], []
    data = df[features + [target]].values
    for i in range(0, len(df) - lookback - 1, stride):
        X.append(data[i:i+lookback, :-1])
        y.append(data[i+lookback, -1])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    return X, y
