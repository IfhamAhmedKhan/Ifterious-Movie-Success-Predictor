from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import streamlit as st


def train_model(df):
    X = df[['num_critic_for_reviews', 'duration', 'num_voted_users', 'num_user_for_reviews', 'movie_facebook_likes', 'director_facebook_likes']]
    df['success_label'] = (df['imdb_score'] > df['imdb_score'].mean()).astype(int)  # Binary label based on IMDb score

    y = df['success_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    return clf

df = pd.read_csv("movie_metadata.csv")

df['gross'].fillna(df['gross'].mean(), inplace=True)
df['budget'].fillna(df['budget'].mean(), inplace=True)
df.dropna(inplace=True)
df['main_genre'] = df['genres'].apply(lambda x: x.split('|')[0] if '|' in x else x)
df['success_label'] = (df['imdb_score'] > df['imdb_score'].mean()).astype(int)  # Binary label based on IMDb score

clf = train_model(df)
