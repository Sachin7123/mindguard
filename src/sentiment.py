import pandas as pd
from textblob import TextBlob
import re


def clean_text(text):
    # Remove URLs, special characters, extra spaces
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@w+|\#", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
    text = text.strip().lower()
    return text


def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # Label based on polarity score
    if polarity > 0.3:
        sentiment = "Positive"
    elif polarity < -0.3:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, polarity


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/reddit_posts.csv")

    # Combine title + text
    df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["clean_text"] = df["full_text"].apply(clean_text)

    # Analyze sentiment
    df[["sentiment", "polarity"]] = df["clean_text"].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

    # Save new version
    df.to_csv("data/reddit_sentiment_labeled.csv", index=False)
    print("✅ Sentiment analysis complete! Saved to reddit_sentiment_labeled.csv")
