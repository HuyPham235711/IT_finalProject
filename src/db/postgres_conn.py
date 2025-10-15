# src/db/postgres_conn.py
from sqlalchemy import create_engine
import pandas as pd

def get_postgres_engine():
    PG_USER = "postgres"
    PG_PASS = "123456789"
    PG_HOST = "localhost"
    PG_PORT = "5432"
    PG_DB   = "postgres"
    conn_str = f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(conn_str)

def load_table_to_df(table_name, schema="it_final", limit=None):
    engine = get_postgres_engine()
    query = f"SELECT * FROM {schema}.{table_name}"
    if limit:
        query += f" LIMIT {limit}"
    return pd.read_sql(query, engine)
