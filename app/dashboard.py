import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# ---------- Load Environment ----------
load_dotenv()
DATA_FILE = "data/reddit_bert_sentiment.csv"


# ---------- Load Data ----------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    return df


df = load_data()

# ---------- Page Configuration ----------
st.set_page_config(page_title="MindGuard – Reddit Sentiment", layout="wide")

st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            text-align: center;
            color: #3D5AFE;
            margin-bottom: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1rem;
            padding: 10px;
            color: #3D5AFE;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 2px solid #3D5AFE;
            background-color: #e8f0fe;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown("<h1>🧠 MindGuard Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Sentiment Analysis for Reddit Mental Health Posts</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------- Sidebar Filters ----------
with st.sidebar:
    st.markdown("## ⚙️ Filters")
    model = st.selectbox("🧠 Select Model", ["TextBlob", "BERT"])
    subs = st.multiselect(
        "📂 Subreddits",
        sorted(df["subreddit"].dropna().unique()),
        default=list(df["subreddit"].dropna().unique()),
    )
    keyword = st.text_input("🔍 Keyword Search")
    if st.button("🔁 Reset Filters"):
        st.experimental_rerun()

# ---------- Filter Data ----------
filtered_df = df[df["subreddit"].isin(subs)]
if keyword:
    filtered_df = filtered_df[
        filtered_df["title"].str.contains(keyword, case=False, na=False)
    ]

# ---------- Sentiment Column Selection ----------
sent_col = "textblob_sentiment" if model == "TextBlob" else "bert_sentiment"
if sent_col not in filtered_df.columns:
    st.error(f"❌ Sentiment column '{sent_col}' not found in dataset.")
    st.stop()

filtered_df[sent_col] = (
    filtered_df[sent_col].astype(str).str.capitalize().fillna("Unknown")
)

# ---------- Count Sentiment Values ----------
total = len(filtered_df)
positive = (filtered_df[sent_col] == "Positive").sum()
negative = (filtered_df[sent_col] == "Negative").sum()
other = total - (positive + negative)
sent_counts = filtered_df[sent_col].value_counts()

# ---------- Custom Summary Cards with Dark Text ----------
st.markdown("### 📊 Dashboard Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
    <div style="background:#f2f4f8;padding:1.5rem;border-radius:12px;text-align:center;box-shadow:0 1px 5px rgba(0,0,0,0.05);">
        <h4 style="color:#1A237E;margin:0;">🧾 Total</h4>
        <p style="font-size:1.6rem;font-weight:700;color:#111;margin:0;">{total:,}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div style="background:#e8f5e9;padding:1.5rem;border-radius:12px;text-align:center;box-shadow:0 1px 5px rgba(0,0,0,0.05);">
        <h4 style="color:#1B5E20;margin:0;">🟢 Positive</h4>
        <p style="font-size:1.6rem;font-weight:700;color:#111;margin:0;">{positive:,}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div style="background:#ffebee;padding:1.5rem;border-radius:12px;text-align:center;box-shadow:0 1px 5px rgba(0,0,0,0.05);">
        <h4 style="color:#B71C1C;margin:0;">🔴 Negative</h4>
        <p style="font-size:1.6rem;font-weight:700;color:#111;margin:0;">{negative:,}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
    <div style="background:#f9fbe7;padding:1.5rem;border-radius:12px;text-align:center;box-shadow:0 1px 5px rgba(0,0,0,0.05);">
        <h4 style="color:#424242;margin:0;">❓ Other</h4>
        <p style="font-size:1.6rem;font-weight:700;color:#111;margin:0;">{other:,}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ---------- Charts ----------
st.markdown(f"### 📈 Sentiment Distribution – {model}")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig, ax = plt.subplots()
    colors = ["#66BB6A", "#FF5252", "#9E9E9E"]
    ax.pie(
        sent_counts,
        labels=sent_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[: len(sent_counts)],
    )
    ax.axis("equal")
    st.pyplot(fig)

with chart_col2:
    st.bar_chart(sent_counts)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📝 Post Preview", "🔍 Model Comparison"])

with tab1:
    st.markdown("### 🗂️ Recent Reddit Posts")
    st.dataframe(
        filtered_df[["title", sent_col, "score", "created_utc", "subreddit"]]
        .sort_values("created_utc", ascending=False)
        .head(25),
        use_container_width=True,
    )

with tab2:
    st.markdown("### 🧠 TextBlob vs BERT")
    st.dataframe(
        filtered_df[
            [
                "title",
                "textblob_sentiment",
                "bert_sentiment",
                "score",
                "created_utc",
                "subreddit",
            ]
        ].head(25),
        use_container_width=True,
    )

# ---------- Download ----------
st.markdown("### ⬇️ Export CSV")
st.download_button(
    label="Download Filtered Results",
    data=filtered_df.to_csv(index=False),
    file_name="mindguard_sentiment_filtered.csv",
    mime="text/csv",
)
