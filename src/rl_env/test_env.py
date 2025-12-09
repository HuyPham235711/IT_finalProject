from src.rl_env.trading_env import TradingEnv
import pandas as pd
from sqlalchemy import create_engine

PG_CONN_STR = "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres"
engine = create_engine(PG_CONN_STR)

# Load close prices
ohlcv_df = pd.read_sql("SELECT close FROM it_final.processed_ohlcv_test ORDER BY datetime", engine)
ohlcv_df = ohlcv_df.tail(3589)

# Create env
env = TradingEnv(
    fusion_emb_path="E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/fusion_rl/fusion_embeddings.npy",
    ohlcv_df=ohlcv_df
)

state = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # random Buy/Sell
    state, reward, done, info = env.step(action)
    if env.current_step % 500 == 0:
        print(info)
