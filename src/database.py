import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load Mongo URI from .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Connect to DB
client = MongoClient(MONGO_URI)
db = client["mindguard"]
collection = db["reddit_posts"]


def insert_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    data = df.to_dict(orient="records")

    result = collection.insert_many(data)
    print(f"✅ Inserted {len(result.inserted_ids)} records into MongoDB.")


if __name__ == "__main__":
    insert_data_from_csv("data/reddit_sentiment_labeled.csv")
