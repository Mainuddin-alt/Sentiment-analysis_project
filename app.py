"""
Twitter Sentiment Analysis - Streamlit App
Page 1: Insights  (dataset overview, model performance, top words)
Page 2: Analyze   (type a tweet, get the sentiment + confidence)

Run:  streamlit run app.py
Needs Twitter_Data.csv in the same folder (or upload it from the sidebar).
"""

import io
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Page config + constants
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "Twitter_Data.csv"

LABELS = {-1: "Negative", 0: "Neutral", 1: "Positive"}
ORDER = ["Negative", "Neutral", "Positive"]
COLORS = {"Negative": "#E5484D", "Neutral": "#8B95A7", "Positive": "#1FA37A"}
EMOJI = {"Negative": "😠", "Neutral": "😐", "Positive": "😊"}
SUMMARY = {
    "Negative": "This tweet sounds critical or unhappy.",
    "Neutral": "This tweet doesn't lean strongly either way.",
    "Positive": "This tweet sounds supportive or happy.",
}

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, button, textarea, input {
    font-family: 'Manrope', system-ui, sans-serif !important;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
#MainMenu, footer { visibility: hidden; }

/* Hero banner */
.hero {
    background: #14213D;
    color: #F5F7FB;
    border-radius: 18px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.6rem;
}
.hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0 0 .4rem 0; color: #F5F7FB; letter-spacing: -0.02em; }
.hero p  { font-size: 1.02rem; margin: 0; color: #B9C3D6; max-width: 640px; line-height: 1.6; }

/* Stat tiles */
.stat {
    border: 1px solid rgba(128,138,160,.28);
    border-left-width: 5px;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    background: rgba(128,138,160,.06);
}
.stat .label { font-size: .85rem; opacity: .75; font-weight: 600; }
.stat .value { font-size: 1.9rem; font-weight: 800; line-height: 1.25; letter-spacing: -0.02em; }

/* Verdict card (prediction page) */
.verdict {
    border-radius: 18px;
    padding: 2rem 1.6rem;
    text-align: center;
    border: 2px solid var(--c);
    background: color-mix(in srgb, var(--c) 12%, transparent);
}
.verdict .emoji { font-size: 3.6rem; line-height: 1; }
.verdict .name  { font-size: 2.4rem; font-weight: 800; color: var(--c); margin: .5rem 0 .2rem; letter-spacing: -0.02em; }
.verdict .conf  { font-size: 1.05rem; font-weight: 600; opacity: .9; }
.verdict .note  { font-size: .95rem; opacity: .7; margin-top: .5rem; }

.empty-box {
    border: 2px dashed rgba(128,138,160,.45);
    border-radius: 18px;
    padding: 3rem 1.5rem;
    text-align: center;
    opacity: .75;
}

/* Sidebar */
section[data-testid="stSidebar"] { border-right: 1px solid rgba(128,138,160,.25); }
.stTabs [data-baseweb="tab"] { font-weight: 700; }
.stButton > button { border-radius: 10px; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Data + model (cached so the app stays fast)
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)  # mentions
    text = re.sub(r"#", "", text)  # hashtag symbol
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # special chars + numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(show_spinner="Cleaning tweets...")
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df["clean_tweet"] = df["clean_text"].apply(clean_text)
    df = df.dropna(subset=["clean_tweet", "category"]).copy()
    df = df[df["category"].isin([-1, 0, 1])].copy()
    df["category"] = df["category"].astype(int)
    df["sentiment"] = df["category"].map(LABELS)
    return df


@st.cache_resource(show_spinner="Training the model (first run only)...")
def train_model(df: pd.DataFrame):
    X = df["clean_tweet"].astype(str)
    y = df["category"].astype(int)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred, target_names=ORDER, output_dict=True
    )
    return {
        "vectorizer": vectorizer,
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
        "report": report,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
    }


def predict(tweet: str, bundle: dict):
    cleaned = clean_text(tweet)
    if not cleaned:
        return None
    vec = bundle["vectorizer"].transform([cleaned])
    probs = bundle["model"].predict_proba(vec)[0]
    classes = bundle["model"].classes_
    prob_map = {LABELS[int(c)]: float(p) for c, p in zip(classes, probs)}
    label = max(prob_map, key=prob_map.get)
    return label, prob_map, cleaned


def stat_tile(label: str, value: str, color: str) -> str:
    return (
        f'<div class="stat" style="border-left-color:{color}">'
        f'<div class="label">{label}</div><div class="value">{value}</div></div>'
    )


def style_fig(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif"),
    )
    return fig


# ----------------------------------------------------------------------
# Load data (file on disk, or uploaded from sidebar)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💬 Tweet Sentiment")
    page = st.radio(
        "Go to",
        ["Insights", "Analyze a tweet"],
        label_visibility="collapsed",
    )
    st.divider()

    uploaded = None
    if not os.path.exists(DATA_FILE):
        uploaded = st.file_uploader("Upload Twitter_Data.csv", type="csv")

file_bytes = None
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        file_bytes = f.read()
elif uploaded is not None:
    file_bytes = uploaded.getvalue()

if file_bytes is None:
    st.markdown(
        """<div class="hero"><h1>Tweet Sentiment Analyzer</h1>
        <p>Add your dataset to get started. Place <b>Twitter_Data.csv</b> next to
        app.py, or upload it from the sidebar.</p></div>""",
        unsafe_allow_html=True,
    )
    st.stop()

df = load_and_clean(file_bytes)
bundle = train_model(df)

with st.sidebar:
    st.caption("Model")
    st.markdown(
        f"**Logistic Regression** on TF-IDF (5,000 words)\n\n"
        f"Accuracy: **{bundle['accuracy'] * 100:.1f}%**\n\n"
        f"Tweets: **{len(df):,}**"
    )


# ======================================================================
# PAGE 1 - INSIGHTS
# ======================================================================
if page == "Insights":
    st.markdown(
        """<div class="hero"><h1>What does Twitter feel?</h1>
        <p>A look at the tweets, how they split across positive, neutral and
        negative sentiment, and how well the model tells them apart.</p></div>""",
        unsafe_allow_html=True,
    )

    counts = df["sentiment"].value_counts().reindex(ORDER).fillna(0).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_tile("Total tweets", f"{len(df):,}", "#2B5FD9"), unsafe_allow_html=True)
    c2.markdown(stat_tile("Positive", f"{counts['Positive']:,}", COLORS["Positive"]), unsafe_allow_html=True)
    c3.markdown(stat_tile("Neutral", f"{counts['Neutral']:,}", COLORS["Neutral"]), unsafe_allow_html=True)
    c4.markdown(stat_tile("Negative", f"{counts['Negative']:,}", COLORS["Negative"]), unsafe_allow_html=True)

    st.write("")
    tab_data, tab_model, tab_words = st.tabs(
        ["📊 Sentiment split", "🎯 Model performance", "🔤 Top words"]
    )

    # ---- Tab 1: distribution -------------------------------------------
    with tab_data:
        left, right = st.columns(2)

        bar = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
            color_discrete_map=COLORS,
            text=[f"{v:,}" for v in counts.values],
            labels={"x": "Sentiment", "y": "Number of tweets"},
            title="Tweets per sentiment",
        )
        bar.update_traces(textposition="outside")
        bar.update_layout(showlegend=False)
        left.plotly_chart(style_fig(bar), use_container_width=True)

        donut = go.Figure(
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.55,
                marker=dict(colors=[COLORS[k] for k in counts.index]),
                sort=False,
                textinfo="percent",
            )
        )
        donut.update_layout(title="Share of each sentiment")
        right.plotly_chart(style_fig(donut), use_container_width=True)

        st.markdown("##### Sample tweets")
        pick = st.selectbox("Show tweets that are", ["All"] + ORDER)
        view = df if pick == "All" else df[df["sentiment"] == pick]
        sample = view[["clean_text", "sentiment"]].sample(
            min(8, len(view)), random_state=None
        )
        sample.columns = ["Tweet", "Sentiment"]
        st.dataframe(sample, use_container_width=True, hide_index=True)

    # ---- Tab 2: performance --------------------------------------------
    with tab_model:
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{bundle['accuracy'] * 100:.2f}%")
        m2.metric("Training tweets", f"{bundle['n_train']:,}")
        m3.metric("Testing tweets", f"{bundle['n_test']:,}")

        left, right = st.columns([1, 1])

        cm = px.imshow(
            bundle["cm"],
            x=ORDER,
            y=ORDER,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Tweets"),
            title="Confusion matrix",
        )
        left.plotly_chart(style_fig(cm, 400), use_container_width=True)

        rep = pd.DataFrame(bundle["report"]).T.drop(index="accuracy", errors="ignore")
        rep = rep[["precision", "recall", "f1-score", "support"]]
        rep["support"] = rep["support"].astype(int)
        right.markdown("**Classification report**")
        right.dataframe(
            rep.style.format(
                {"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}
            ),
            use_container_width=True,
        )
        right.caption(
            "Precision: how often a predicted label is right. "
            "Recall: how many real tweets of that label were found."
        )

    # ---- Tab 3: top words ----------------------------------------------
    with tab_words:
        st.caption("Words that push the model most strongly toward each sentiment.")
        feature_names = np.array(bundle["vectorizer"].get_feature_names_out())
        coefs = bundle["model"].coef_
        classes = bundle["model"].classes_

        cols = st.columns(3)
        for col, name in zip(cols, ORDER):
            idx = list(classes).index({v: k for k, v in LABELS.items()}[name])
            top = np.argsort(coefs[idx])[-10:]
            fig = go.Figure(
                go.Bar(
                    x=coefs[idx][top],
                    y=feature_names[top],
                    orientation="h",
                    marker_color=COLORS[name],
                )
            )
            fig.update_layout(title=name, xaxis_title="Weight")
            col.plotly_chart(style_fig(fig, 380), use_container_width=True)


# ======================================================================
# PAGE 2 - ANALYZE A TWEET
# ======================================================================
else:
    st.markdown(
        """<div class="hero"><h1>Analyze a tweet</h1>
        <p>Paste or type any tweet and see whether it reads as positive,
        neutral or negative, along with how confident the model is.</p></div>""",
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("tweet", "")
    st.session_state.setdefault("history", [])

    st.session_state.setdefault("guess", "Not sure")

    def load_random(category: str):
        """Fill the text box with a real tweet of the chosen category."""
        pool = df.loc[df["sentiment"] == category, "clean_text"]
        st.session_state["tweet"] = str(pool.sample(1).iloc[0])
        st.session_state["guess"] = category

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.radio(
            "What do you think this tweet is?",
            ["Not sure", "Positive", "Neutral", "Negative"],
            key="guess",
            horizontal=True,
            help="Pick a category, then see if the model agrees with you.",
        )

        st.text_area(
            "Your tweet",
            key="tweet",
            height=150,
            placeholder="Type or paste a tweet here...",
        )

        st.caption("No tweet handy? Load a real one from the dataset")
        e1, e2, e3 = st.columns(3)
        for col, name in zip((e1, e2, e3), ("Positive", "Neutral", "Negative")):
            col.button(
                f"{EMOJI[name]} {name}",
                on_click=load_random,
                args=(name,),
                use_container_width=True,
                key=f"ex_{name}",
            )

        go_btn = st.button("Analyze sentiment", type="primary", use_container_width=True)

    with right:
        tweet = st.session_state["tweet"]
        result = predict(tweet, bundle) if tweet.strip() else None

        if go_btn and not tweet.strip():
            st.warning("Enter a tweet first, then press Analyze sentiment.")

        if result is None:
            st.markdown(
                """<div class="empty-box"><div style="font-size:2.4rem">💬</div>
                <b>No result yet</b><br>Write a tweet or pick an example to see its sentiment.</div>""",
                unsafe_allow_html=True,
            )
        else:
            label, probs, cleaned = result
            st.markdown(
                f"""<div class="verdict" style="--c:{COLORS[label]}">
                <div class="emoji">{EMOJI[label]}</div>
                <div class="name">{label}</div>
                <div class="conf">{probs[label] * 100:.1f}% confident</div>
                <div class="note">{SUMMARY[label]}</div></div>""",
                unsafe_allow_html=True,
            )

            guess = st.session_state["guess"]
            if guess != "Not sure":
                if guess == label:
                    st.success(f"The model agrees with you: {label}.")
                else:
                    st.warning(
                        f"You picked {guess}, but the model reads this as {label}."
                    )

            if go_btn:
                hist = st.session_state["history"]
                if not hist or hist[0]["Tweet"] != tweet:
                    hist.insert(0, {"Tweet": tweet, "Your guess": st.session_state["guess"], "Model says": label, "Confidence": f"{probs[label] * 100:.1f}%"})
                    del hist[10:]

    # Probability breakdown + details
    if result is not None:
        label, probs, cleaned = result
        st.write("")
        b1, b2 = st.columns([1.15, 1], gap="large")

        with b1:
            fig = go.Figure(
                go.Bar(
                    x=[probs[k] * 100 for k in ORDER],
                    y=ORDER,
                    orientation="h",
                    marker_color=[COLORS[k] for k in ORDER],
                    text=[f"{probs[k] * 100:.1f}%" for k in ORDER],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Confidence for each sentiment",
                xaxis=dict(range=[0, 115], title="Probability (%)"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)

        with b2:
            st.markdown("**How the model saw your tweet**")
            st.caption("After removing links, mentions, numbers and symbols:")
            st.code(cleaned or "(empty)", language=None)

    # History
    if st.session_state["history"]:
        st.markdown("##### Recent analyses")
        st.dataframe(
            pd.DataFrame(st.session_state["history"]),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()