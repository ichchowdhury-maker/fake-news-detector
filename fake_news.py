import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

df = pd.read_csv("news.csv")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

accuracy = model.score(X_test_vec, y_test)

st.title("Fake News Detector (PRO VERSION)")
st.write(f"Model Accuracy: {accuracy:.2f}")

news = st.text_area("Enter News Text")

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter news text")
    else:
        news_vec = vectorizer.transform([news])
        prediction = model.predict(news_vec)

        if prediction[0] == 1:
            st.success("Real News")
        else:
            st.error("Fake News")