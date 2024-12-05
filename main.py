import streamlit as st
import requests
import pandas as pd
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration
class Config:
    """Application configuration settings."""

    AHREFS_API_URL = "https://api.ahrefs.com/v3/serp-overview/serp-overview"
    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

    # API Keys
    AHREFS_AUTH_TOKEN = os.getenv("AHREFS_AUTH_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# API Interaction Utilities
class APIClient:
    """Handles API interactions for different services."""

    @staticmethod
    def fetch_ahrefs_serp_data(
        keyword: str,
        country: str,
        select: str,
        date: Optional[str] = None,
        top_positions: Optional[int] = None,
    ) -> Optional[Dict]:
        params = {
            "country": country,
            "keyword": keyword,
            "select": select,
            "date": date if date else None,
            "top_positions": top_positions if top_positions else None,
        }
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {Config.AHREFS_AUTH_TOKEN}",
        }

        try:
            response = requests.get(
                Config.AHREFS_API_URL, headers=headers, params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching Ahrefs data: {e}")
            return None

    @staticmethod
    def analyze_sentiment_with_openai(title: str) -> Dict:
        if not title or not title.strip():
            return {
                "sentiment": "neutral",
                "sentiment_score": 0,
                "primary_emotion": "none",
                "emotions": [],
                "emotional_intensity": 0,
                "intent": "none",
                "tone": "none",
                "key_psychological_triggers": [],
                "brief_explanation": "No title provided",
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        }

        payload = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an advanced sentiment and emotion analysis AI. Your task is to provide a precise, structured JSON analysis of the given text. "
                        "REQUIRED JSON STRUCTURE:\n"
                        "{\n"
                        "  \"sentiment\": \"string (exact: 'very_positive'|'positive'|'neutral'|'negative'|'very_negative')\",\n"
                        '  "sentiment_score": "number (float between -1.0 and 1.0)",\n'
                        '  "primary_emotion": "string (most dominant emotion)",\n'
                        '  "emotions": "array of strings (max 3 emotions)",\n'
                        '  "emotional_intensity": "number (float between 0 and 1.0)",\n'
                        "  \"intent\": \"string (one of: 'informational'|'transactional'|'navigational'|'commercial')\",\n"
                        "  \"tone\": \"string (one of: 'professional'|'conversational'|'technical'|'persuasive'|'urgent'|'empathetic')\",\n"
                        '  "key_psychological_triggers": "array of strings (max 3 triggers)",\n'
                        '  "brief_explanation": "string (max 100 characters explaining the analysis)"\n'
                        "}\n\n"
                        "ANALYSIS GUIDELINES:\n"
                        "1. Sentiment Analysis:\n"
                        "- Precisely categorize sentiment\n"
                        "- Provide nuanced sentiment score\n"
                        "- Consider linguistic context and subtle implications\n\n"
                        "2. Emotion Detection:\n"
                        "- Identify dominant and secondary emotions\n"
                        "- Assess emotional intensity\n"
                        "- Consider psychological undertones\n\n"
                        "3. Additional Insights:\n"
                        "- Determine precise user intent\n"
                        "- Analyze communication tone\n"
                        "- Identify psychological triggers\n\n"
                        "CRITICAL INSTRUCTIONS:\n"
                        "- ALWAYS return valid JSON\n"
                        "- Use only specified enum values\n"
                        "- Be concise but insightful\n"
                        "- Prioritize accuracy over complexity"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Analyze the sentiment and emotions in this title: '{title}'",
                },
            ],
            "temperature": 0.5,
            "max_tokens": 300,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }

        try:
            response = requests.post(
                Config.OPENAI_API_URL, headers=headers, json=payload
            )
            response.raise_for_status()

            # Parse the response
            analysis = response.json()
            if "choices" in analysis and len(analysis["choices"]) > 0:
                sentiment_data = json.loads(
                    analysis["choices"][0]["message"]["content"]
                )

                # Extract and ensure that we have default return values
                return {
                    "sentiment": sentiment_data.get("sentiment", "neutral"),
                    "sentiment_score": sentiment_data.get("sentiment_score", 0),
                    "primary_emotion": sentiment_data.get("primary_emotion", "none"),
                    "emotions": sentiment_data.get("emotions", []),
                    "emotional_intensity": sentiment_data.get("emotional_intensity", 0),
                    "intent": sentiment_data.get("intent", "none"),
                    "tone": sentiment_data.get("tone", "none"),
                    "key_psychological_triggers": sentiment_data.get(
                        "key_psychological_triggers", []
                    ),
                    "brief_explanation": sentiment_data.get(
                        "brief_explanation", "No explanation provided"
                    ),
                }
            else:
                st.error("OpenAI returned an empty analysis result.")
                return {
                    "sentiment": "neutral",
                    "sentiment_score": 0,
                    "primary_emotion": "none",
                    "emotions": [],
                    "emotional_intensity": 0,
                    "intent": "none",
                    "tone": "none",
                    "key_psychological_triggers": [],
                    "brief_explanation": "OpenAI analysis result was empty.",
                }

        except (
            requests.exceptions.RequestException,
            json.JSONDecodeError,
            KeyError,
        ) as e:
            return {
                "sentiment": "error",
                "sentiment_score": 0,
                "primary_emotion": "none",
                "emotions": [],
                "emotional_intensity": 0,
                "intent": "none",
                "tone": "none",
                "key_psychological_triggers": [],
                "brief_explanation": f"Error: {str(e)}",
            }


# Data Processing Utilities
class DataProcessor:
    """Handles data processing and analysis."""

    @staticmethod
    def enrich_serp_data_with_sentiment(positions: List[Dict]) -> List[Dict]:
        enriched_positions = []
        for pos in positions:
            title = pos.get("title", "")
            sentiment_analysis = APIClient.analyze_sentiment_with_openai(title)
            enriched_pos = {**pos, **sentiment_analysis}
            enriched_positions.append(enriched_pos)
        return enriched_positions

    @staticmethod
    def generate_insights(enriched_data: List[Dict]) -> Dict:
        if not enriched_data:
            return {"error": "No data available for insights generation"}

        # Sentiment distribution
        sentiment_counts = {
            "positive": sum(
                1 for item in enriched_data if item.get("sentiment") == "positive"
            ),
            "negative": sum(
                1 for item in enriched_data if item.get("sentiment") == "negative"
            ),
            "neutral": sum(
                1 for item in enriched_data if item.get("sentiment") == "neutral"
            ),
        }

        # Emotion analysis
        all_emotions = [
            emotion for item in enriched_data for emotion in item.get("emotions", [])
        ]
        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        return {
            "sentiment_distribution": sentiment_counts,
            "top_emotions": sorted(
                emotion_counts.items(), key=lambda x: x[1], reverse=True
            )[:3],
            "average_sentiment_score": (
                sum(item.get("sentiment_score", 0) for item in enriched_data)
                / len(enriched_data)
                if enriched_data
                else 0
            ),
            "total_positions": len(enriched_data),
        }


# Streamlit App
def main():
    st.set_page_config(
        page_title="Keyword Sentiment Analysis Tool",
        page_icon="🔍",
        layout="wide",
    )

    st.title("📈 Keyword Sentiment Analysis Tool")
    st.markdown(
        """
    ### Discover the Sentiment and Emotional Landscape of Keywords
    
    Analyze keyword sentiment using AI-driven insights.
    """
    )

    st.sidebar.header("🛠 Analysis Parameters")

    keyword = st.sidebar.text_input(
        "Keyword",
        value="digital marketing",
        help="Enter the keyword to analyze in-depth",
    )

    country = st.sidebar.selectbox(
        "Country",
        options=["us", "gb", "ca", "au", "in"],
        index=0,
        help="Select the target country for analysis",
    )

    top_positions = st.sidebar.slider(
        "Top Positions to Analyze",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of top results to process",
    )

    # Analysis trigger
    if st.sidebar.button("🚀 Analyze Sentiment"):
        # Validate API keys
        if not Config.AHREFS_AUTH_TOKEN or not Config.OPENAI_API_KEY:
            st.error(
                "❌ Missing API credentials. Please set AHREFS_AUTH_TOKEN and OPENAI_API_KEY in your .env file."
            )
            return

        # Initialize progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Fetch and process data
        with st.spinner("🔬 Analyzing data and sentiment..."):
            # Fetch data
            serp_data = APIClient.fetch_ahrefs_serp_data(
                keyword,
                country,
                "position,title,traffic,url",
                top_positions=top_positions,
            )

            if serp_data and "positions" in serp_data:
                # Update progress
                progress_text.text("🔍 Fetching data...")
                progress_bar.progress(20)

                # Enrich data with sentiment
                enriched_positions = []
                total_positions = len(serp_data.get("positions", []))
                for i, pos in enumerate(serp_data.get("positions", [])):
                    title = pos.get("title", "")
                    sentiment_analysis = APIClient.analyze_sentiment_with_openai(title)
                    enriched_pos = {**pos, **sentiment_analysis}
                    enriched_positions.append(enriched_pos)

                    # Update progress within the loop
                    progress_percentage = 20 + int(((i + 1) / total_positions) * 70)
                    progress_percentage = min(max(progress_percentage, 0), 100)
                    progress_bar.progress(progress_percentage)
                    progress_text.text(
                        f"🔎 Analyzing sentiment for position {i + 1}/{total_positions}..."
                    )

                # Generate insights
                progress_text.text("📈 Generating insights...")
                insights = DataProcessor.generate_insights(enriched_positions)
                progress_bar.progress(100)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Sentiment Distribution")
                    st.bar_chart(insights["sentiment_distribution"])

                with col2:
                    st.subheader("🎭 Top Detected Emotions")
                    emotion_df = pd.DataFrame(
                        insights["top_emotions"], columns=["Emotion", "Frequency"]
                    )
                    st.bar_chart(emotion_df.set_index("Emotion"))

                # Detailed results
                st.subheader("🔍 Detailed Sentiment Analysis")
                df = pd.DataFrame(enriched_positions)
                st.dataframe(df)

                # CSV Export
                export_filename = f"sentiment_analysis_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    label="📥 Export CSV",
                    data=df.to_csv(index=False),
                    file_name=export_filename,
                    mime='text/csv'
                )

            else:
                st.warning("❗ No data retrieved. Please check your parameters.")

        # Clear progress indicators
        progress_bar.empty()
        progress_text.empty()


if __name__ == "__main__":
    main()