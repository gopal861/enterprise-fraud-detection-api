import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

CSV_PATH = r"C:\Users\IP\Desktop\fraud_project\data\creditcard.csv"


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def insert_batch(df: pd.DataFrame):
    conn = get_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO transactions (
            transaction_time_seconds, amount,
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28,
            is_fraud
        )
        VALUES (
            %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s
        );
    """

    for _, row in df.iterrows():
        cursor.execute(
            insert_query,
            (
                int(row["Time"]),
                float(row["Amount"]),
                *[float(row[f"V{i}"]) for i in range(1, 29)],
                bool(row["Class"]),
            ),
        )

    conn.commit()
    cursor.close()
    conn.close()


def main():
    df = load_csv(CSV_PATH)
    insert_batch(df)
    print("Full dataset inserted successfully.")


if __name__ == "__main__":
    main()