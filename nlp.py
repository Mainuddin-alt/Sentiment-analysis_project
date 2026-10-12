# Step 1: Import Libraries

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# Step 2: Load Dataset
# ============================================================

df = pd.read_csv("Twitter_Data.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# Step 3: Check Dataset Information
# ============================================================

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# Step 4: Clean Tweets
# ============================================================

def clean_text(text):
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    
    # Remove hashtag symbol
    text = re.sub(r"#", "", text)
    
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


df["clean_tweet"] = df["clean_text"].apply(clean_text)

print("\nOriginal and Cleaned Tweets:")
print(df[["clean_text", "clean_tweet"]].head())


# ============================================================
# Step 5: Remove Missing and Invalid Data
# ============================================================

df = df.dropna(subset=["clean_tweet", "category"]).copy()

# Keep only valid sentiment categories
df = df[df["category"].isin([-1, 0, 1])].copy()

print("\nDataset Shape After Cleaning:", df.shape)

print("\nSentiment Count:")
print(df["category"].value_counts())


# ============================================================
# Step 6: Convert Category into Sentiment Names
# ============================================================

sentiment_names = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

df["sentiment"] = df["category"].map(sentiment_names)

print("\nSentiment Examples:")
print(df[["clean_tweet", "category", "sentiment"]].head())


# ============================================================
# Step 7: Basic Bar Graph
# ============================================================

sentiment_count = df["sentiment"].value_counts()

plt.figure(figsize=(7, 5))

sns.barplot(
    x=sentiment_count.index,
    y=sentiment_count.values
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")

plt.show()


# ============================================================
# Step 8: Basic Pie Chart
# ============================================================

plt.figure(figsize=(6, 6))

plt.pie(
    sentiment_count.values,
    labels=sentiment_count.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sentiment Distribution")

plt.show()


# ============================================================
# Step 9: Prepare X and Y
# ============================================================

X = df["clean_tweet"].astype(str)

y = df["category"].astype(int)

print("\nX Shape:", X.shape)
print("Y Shape:", y.shape)

print("Missing Values in Y:", y.isnull().sum())


# ============================================================
# Step 10: Convert Text into TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(X)

print("\nTF-IDF Shape:", X_tfidf.shape)


# ============================================================
# Step 11: Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# Step 12: Train Logistic Regression Model
# ============================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# ============================================================
# Step 13: Predict Test Data
# ============================================================

y_pred = model.predict(X_test)

print("\nFirst 20 Predictions:")
print(y_pred[:20])


# ============================================================
# Step 14: Calculate Accuracy
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")


# ============================================================
# Step 15: Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"]
)

plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.title("Confusion Matrix")

plt.show()


# ============================================================
# Step 16: Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Neutral", "Positive"]
    )
)


# ============================================================
# Step 17: User Input Sentiment Prediction
# ============================================================

sentiment_result = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

print("\n========================================")
print("       TWITTER SENTIMENT ANALYSIS")
print("========================================")

while True:

    user_tweet = input(
        "\nEnter a tweet "
        "(or type 'exit' to stop): "
    )

    # Stop program
    if user_tweet.lower() == "exit":
        print("\nProgram stopped.")
        break

    # Clean user input
    cleaned_tweet = clean_text(user_tweet)

    # Convert user input to TF-IDF
    tweet_tfidf = vectorizer.transform([cleaned_tweet])

    # Predict sentiment
    prediction = model.predict(tweet_tfidf)[0]

    # Display result
    print("\nTweet:", user_tweet)
    print(
        "Predicted Sentiment:",
        sentiment_result[int(prediction)]
    )