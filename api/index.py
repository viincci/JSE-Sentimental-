from flask import Flask, jsonify, render_template
import yfinance as yf
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Fetch JSE Top 40 Index data
jse_index = "JSE.JO"
data = yf.download(jse_index, period="7d", interval="1d")

# Get API key from environment variables
API_KEY = os.getenv("NEWS_API_KEY")

# Fetch news data
NEWS_API_URL = f"https://gnews.io/api/v4/search?q=JSE&lang=en&country=za&token={API_KEY}"
response = requests.get(NEWS_API_URL)
news_data = response.json()
articles = news_data.get("articles", [])
news_headlines = [article["title"] for article in articles]

# Load sentiment model
sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
sentiments = [sentiment_pipeline(headline)[0] for headline in news_headlines]

# Convert to DataFrame
sentiment_df = pd.DataFrame({
    "Headline": news_headlines,
    "Sentiment": [s["label"] for s in sentiments],
    "Score": [s["score"] for s in sentiments]
})

# Create sentiment plot
def plot_sentiment():
    sentiment_counts = sentiment_df["Sentiment"].value_counts()
    plt.figure(figsize=(8,5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm")
    plt.title("JSE News Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig("../static/sentiment_plot.png")

plot_sentiment()

@app.route('/')
def index():
    return render_template('index.html', sentiment_data=sentiment_df.to_dict(orient='records'))

@app.route('/api/sentiment')
def get_sentiment():
    return jsonify(sentiment_df.to_dict(orient='records'))

# Vercel handler
def handler(event, context):
    return app(event, context)
