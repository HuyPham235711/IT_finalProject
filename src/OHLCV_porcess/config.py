DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "123456789",
    "host": "localhost",
    "port": "5432"
}

BATCH_SIZE = 1000

# định nghĩa các bảng dữ liệu gốc và bảng đích tương ứng
TABLES = {
    "train": (
        "it_final.ohlcv_train",
        "it_final.processed_ohlcv_train"
    ),
    "valid": (
        "it_final.ohlcv_valid",
        "it_final.processed_ohlcv_valid"
    ),
    "test": (
        "it_final.ohlcv_test",
        "it_final.processed_ohlcv_test"
    ),
    "backtest": (
        "it_final.ohlcv_backtest",
        "it_final.processed_ohlcv_backtest"
    ),
}

# file lưu thông tin scaler (min/max)
SCALER_PATH = "scaler.json"