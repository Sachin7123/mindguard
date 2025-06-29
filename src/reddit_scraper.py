import praw
import pandas as pd
from dotenv import load_dotenv
import os

# Load credentials
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)


def scrape_subreddit(subreddit_name, post_limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for post in subreddit.hot(limit=post_limit):
        if not post.stickied:
            posts.append(
                {
                    "title": post.title,
                    "text": post.selftext,
                    "score": post.score,
                    "url": post.url,
                    "created_utc": post.created_utc,
                    "subreddit": subreddit_name,
                }
            )

    df = pd.DataFrame(posts)
    return df


if __name__ == "__main__":
    # Customize here
    subreddits = ["depression", "anxiety"]
    all_posts = pd.DataFrame()

    for sub in subreddits:
        df = scrape_subreddit(sub, post_limit=100)
        all_posts = pd.concat([all_posts, df], ignore_index=True)

    all_posts.to_csv("data/reddit_posts.csv", index=False)
    print(f"✅ Scraped {len(all_posts)} posts and saved to data/reddit_posts.csv")
