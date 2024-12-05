import streamlit as st
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Embed your Ahrefs API token here
API_TOKEN = "gy0w35-eaNFP6v6vgrb_GIs6cp13SNG70Tk0TdPk"

# Function to fetch top 10 organic results using Ahrefs API
def fetch_organic_results(keyword):
    url = "https://api.ahrefs.com/v3/serp-overview/serp-overview"
    headers = {
        "Authorization": "Bearer gy0w35-eaNFP6v6vgrb_GIs6cp13SNG70Tk0TdPk",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }
    params = {
        "q": keyword,
        "limit": 10
        # "token": "gy0w35-eaNFP6v6vgrb_GIs6cp13SNG70Tk0TdPk"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('organic', [])
    else:
        st.error(f"Error fetching organic results: {response.status_code}")
        st.write("Response Body (Debug):", response.text)
        return []

# Function to perform sentiment analysis
def analyze_sentiment(text):
    return sia.polarity_scores(text)

# Streamlit App
def main():
    st.title("Keyword Sentiment Analysis Tool")

    # Step 1: User Input
    keyword = st.text_input("Enter a keyword:")
    user_url = st.text_input("Enter your URL:")

    if st.button("Analyze"):
        if not keyword or not user_url:
            st.error("Please fill in both fields!")
            return

        # Step 2: Fetch Top 10 Organic Results
        st.write(f"Fetching top 10 organic results for keyword: {keyword}")
        organic_results = fetch_organic_results(keyword)

        if not organic_results:
            st.warning("No results found.")
            return

        # Step 3: Perform Sentiment Analysis
        st.write("Performing sentiment analysis on the results...")
        results = []
        for result in organic_results:
            url = result.get("url", "N/A")
            title = result.get("title", "No Title")
            sentiment = analyze_sentiment(title)
            results.append({"URL": url, "Title": title, "Sentiment": sentiment})

        # Step 4: Display Results
        st.subheader("Sentiment Analysis Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

if __name__ == "__main__":
    main()
