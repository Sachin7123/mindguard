from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from scipy.special import softmax

# Load general-purpose sentiment model
MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def analyze_sentiment(text):
    text = text.strip().replace("\n", " ")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits.numpy()[0])

    labels = ["Negative", "Positive"]
    sentiment = labels[probs.argmax()]
    confidence = probs.max()

    return sentiment, round(confidence, 3)


if __name__ == "__main__":
    df = pd.read_csv("data/reddit_sentiment_labeled.csv")
    df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    results = df["full_text"].apply(lambda x: pd.Series(analyze_sentiment(x)))
    results.columns = ["bert_sentiment", "bert_confidence"]

    df = pd.concat([df, results], axis=1)
    df.rename(
        columns={"sentiment": "textblob_sentiment", "polarity": "textblob_polarity"},
        inplace=True,
    )

    df.to_csv("data/reddit_bert_sentiment.csv", index=False)
    print("✅ Reddit sentiment analysis (BERT) complete and saved.")
