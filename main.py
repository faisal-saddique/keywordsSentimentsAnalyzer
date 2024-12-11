from bs4 import BeautifulSoup
import streamlit as st
import requests
import pandas as pd
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import os
from datetime import datetime
import html2text

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
    def analyze_sentiment_with_openai(text: str) -> Dict:
        if not text or not text.strip():
            return {
                "sentiment": "neutral",
                "sentiment_score": 0,
                "primary_emotion": "none",
                "emotions": [],
                "emotional_intensity": 0,
                "intent": "none",
                "tone": "none",
                "key_psychological_triggers": [],
                "brief_explanation": "No text provided",
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        }

        payload = {
            "model": "gpt-4o",
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
                    "content": f"Analyze the sentiment and emotions in this text: '{text}'",
                },
            ],
            "temperature": 0.5,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
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

    @staticmethod
    def fetch_webpage_content(url: str) -> Optional[str]:
        if not url or not url.strip():
            st.warning("No URL provided.")
            return None

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            html_content = response.text

            # Parse HTML and extract body content
            soup = BeautifulSoup(html_content, 'html.parser')
            body_content = soup.body

            # Remove base64 images and links
            for img in body_content.find_all('img', src=True):
                if img['src'].startswith('data:image'):
                    img.decompose()

            for a in body_content.find_all('a', href=True):
                a.decompose()

            # Convert remaining HTML to markdown text
            markdown_converter = html2text.HTML2Text()
            markdown_converter.ignore_links = True
            markdown_converter.ignore_images = True
            text_content = markdown_converter.handle(str(body_content))

            return text_content

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching webpage content: {e}")
            return None

    @staticmethod
    def generate_summary_and_recommendation(keyword_sentiment: str, webpage_sentiment: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        }

        prompt = (
            "You are an expert in sentiment analysis and content optimization. Based on the following data, provide a concise summary and recommendation. "
            "The data includes the overall sentiment of a keyword and the sentiment of a webpage. Use this format: "
            "'The overall sentiment for the keyword is {keyword_sentiment}, but the webpage sentiment is {webpage_sentiment}. Recommendation: {suggestion}.'\n\n"
            "Data:\n"
            f"- Keyword Sentiment: {keyword_sentiment}\n"
            f"- Webpage Sentiment: {webpage_sentiment}\n\n"
            "Consider the following when crafting your recommendation:\n"
            "- If the webpage sentiment is less positive than the keyword sentiment, suggest ways to make the webpage more positive.\n"
            "- If the webpage sentiment is more positive, suggest maintaining or enhancing the positive aspects.\n"
            "- Provide actionable insights for content improvement."
        )

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 1024,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }

        try:
            response = requests.post(
                Config.OPENAI_API_URL, headers=headers, json=payload
            )
            response.raise_for_status()
            analysis = response.json()
            if "choices" in analysis and len(analysis["choices"]) > 0:
                return analysis["choices"][0]["message"]["content"].strip()
            else:
                return "Unable to generate a summary and recommendation."
        except requests.exceptions.RequestException as e:
            return f"Error generating summary: {e}"


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
        sentiments = [item.get("sentiment") for item in enriched_data]
        sentiment_counts = pd.Series(sentiments).value_counts().to_dict()

        # Emotion analysis
        all_emotions = [
            emotion for item in enriched_data for emotion in item.get("emotions", [])
        ]
        emotion_counts = pd.Series(all_emotions).value_counts().to_dict()

        # Average sentiment score
        sentiment_scores = [
            item.get("sentiment_score", 0)
            for item in enriched_data
            if isinstance(item.get("sentiment_score", 0), (int, float))
        ]
        average_sentiment_score = (
            sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        )

        return {
            "sentiment_distribution": sentiment_counts,
            "top_emotions": sorted(
                emotion_counts.items(), key=lambda x: x[1], reverse=True
            )[:3],
            "average_sentiment_score": average_sentiment_score,
            "total_positions": len(enriched_data),
        }

    def compare_with_webpage(
        enriched_positions: List[Dict], webpage_analysis: Dict, keyword: str
    ) -> Dict:
        """Compare SERP data analysis with webpage content analysis."""
        if not enriched_positions or not webpage_analysis:
            return {"error": "Insufficient data for comparison"}

        # Prepare data for comparison
        serp_sentiment_scores = [
            item.get("sentiment_score", 0)
            for item in enriched_positions
            if isinstance(item.get("sentiment_score", 0), (int, float))
        ]
        serp_average_sentiment = (
            sum(serp_sentiment_scores) / len(serp_sentiment_scores)
            if serp_sentiment_scores
            else 0
        )

        # Keyword density in SERP titles
        serp_keyword_density = sum(
            keyword.lower() in item.get("title", "").lower() for item in enriched_positions
        ) / len(enriched_positions)

        # Keyword density in webpage content
        webpage_content = webpage_analysis.get("content", "")
        if webpage_content.strip():
            webpage_keyword_density = webpage_content.lower().count(keyword.lower()) / len(webpage_content.split())
        else:
            webpage_keyword_density = 0

        comparison = {
            "serp_average_sentiment_score": serp_average_sentiment,
            "webpage_sentiment_score": webpage_analysis.get("sentiment_score", 0),
            "serp_primary_emotions": [
                item.get("primary_emotion", "none") for item in enriched_positions
            ],
            "webpage_primary_emotion": webpage_analysis.get("primary_emotion", "none"),
            "serp_tones": [item.get("tone", "none") for item in enriched_positions],
            "webpage_tone": webpage_analysis.get("tone", "none"),
            "serp_keyword_density": serp_keyword_density,
            "webpage_keyword_density": webpage_keyword_density,
            # Additional comparisons can be added here
        }

        return comparison

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

    # Optional URL input for comparison
    url = st.sidebar.text_input(
        "Webpage URL (Optional)",
        value="",
        help="Enter a URL to compare its content with the keyword analysis",
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
            # Step 1: Fetch SERP data
            progress_text.text("🔍 Fetching SERP data...")
            serp_data = APIClient.fetch_ahrefs_serp_data(
                keyword,
                country,
                "position,title,traffic,url",
                top_positions=top_positions,
            )

            if serp_data and "positions" in serp_data:
                # Update progress
                progress_bar.progress(20)

                # Step 2: Enrich SERP data with sentiment
                enriched_positions = []
                total_positions = len(serp_data.get("positions", []))
                for i, pos in enumerate(serp_data.get("positions", [])):
                    title = pos.get("title", "")
                    sentiment_analysis = APIClient.analyze_sentiment_with_openai(title)
                    enriched_pos = {**pos, **sentiment_analysis}
                    enriched_positions.append(enriched_pos)

                    # Update progress within the loop
                    progress_percentage = 20 + int(((i + 1) / total_positions) * 40)
                    progress_percentage = min(max(progress_percentage, 0), 100)
                    progress_bar.progress(progress_percentage)
                    progress_text.text(
                        f"🔎 Analyzing sentiment for position {i + 1}/{total_positions}..."
                    )

                # Step 3: Generate insights from SERP data
                progress_text.text("📈 Generating insights from SERP data...")
                serp_insights = DataProcessor.generate_insights(enriched_positions)
                progress_bar.progress(65)
            else:
                st.warning("❗ No SERP data retrieved. Please check your parameters.")
                return

            # Step 4: If URL is provided, fetch and analyze webpage content
            if url:
                progress_text.text("🌐 Fetching webpage content...")
                webpage_content = APIClient.fetch_webpage_content(url)
                if webpage_content:
                    progress_bar.progress(70)

                    # Analyze webpage content sentiment
                    progress_text.text("🧠 Analyzing webpage content sentiment...")
                    # st.success(webpage_content[:400])
                    webpage_analysis = APIClient.analyze_sentiment_with_openai(
                        webpage_content
                    )
                    progress_bar.progress(85)
                else:
                    st.warning("❗ Failed to fetch or parse the webpage content.")
                    webpage_analysis = None
            else:
                webpage_analysis = None

            # Step 5: Compare SERP data with webpage analysis
            if webpage_analysis:
                progress_text.text("🔗 Comparing SERP data with webpage analysis...")
                comparison_results = DataProcessor.compare_with_webpage(
                    enriched_positions, webpage_analysis, keyword
                )
                progress_bar.progress(95)
            else:
                comparison_results = None
                progress_bar.progress(95)

            # Complete progress
            progress_bar.progress(100)
            progress_text.text("✅ Analysis complete!")

            # Detailed results
            st.subheader("🔍 Detailed SERP Sentiment Analysis")
            df = pd.DataFrame(enriched_positions)
            st.dataframe(df)
            
            # Display results
            st.subheader("📊 Sentiment Distribution in SERP Results")
            sentiment_df = pd.DataFrame.from_dict(
                serp_insights["sentiment_distribution"],
                orient="index",
                columns=["Count"],
            )
            st.bar_chart(sentiment_df)

            st.subheader("🎭 Top Detected Emotions in SERP Results")
            emotion_df = pd.DataFrame(
                serp_insights["top_emotions"], columns=["Emotion", "Frequency"]
            )
            st.bar_chart(emotion_df.set_index("Emotion"))

            if webpage_analysis:
                st.subheader("🌐 Webpage Content Analysis")
                # Convert the webpage_analysis dictionary to a DataFrame for better display
                webpage_df = pd.DataFrame([webpage_analysis])
                st.table(webpage_df)

                # Additional comparisons
                st.subheader("🔄 Comparison Between SERP and Webpage Content")
                comparison_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Average Sentiment Score",
                            "Primary Emotion",
                            "Tone",
                        ],
                        "SERP Results": [
                            serp_insights["average_sentiment_score"],
                            pd.Series(
                                comparison_results["serp_primary_emotions"]
                            ).mode()[0],
                            pd.Series(comparison_results["serp_tones"]).mode()[0],
                        ],
                        "Webpage Content": [
                            comparison_results["webpage_sentiment_score"],
                            comparison_results["webpage_primary_emotion"],
                            comparison_results["webpage_tone"],
                        ],
                    }
                )
                st.table(comparison_df)

                # Example visualization: Compare sentiment scores
                st.subheader("🔄 Sentiment Score Comparison")
                comparison_data = {
                    "Metric": ["Sentiment Score"],
                    "SERP Results": [serp_insights["average_sentiment_score"]],
                    "Webpage Content": [webpage_analysis.get("sentiment_score", 0)],
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.bar_chart(comparison_df.set_index("Metric"))

                # Generate AI summary and recommendation
                keyword_sentiment = pd.Series(comparison_results["serp_primary_emotions"]).mode()[0]
                webpage_sentiment = comparison_results["webpage_primary_emotion"]
                summary = APIClient.generate_summary_and_recommendation(keyword_sentiment, webpage_sentiment)

                # Display AI summary
                st.subheader("📝 AI Summary and Recommendation")
                st.markdown(summary)

            # CSV Export
            export_filename = f"sentiment_analysis_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button(
                label="📥 Export CSV",
                data=df.to_csv(index=False),
                file_name=export_filename,
                mime="text/csv",
            )

        # Clear progress indicators
        progress_bar.empty()
        progress_text.empty()


if __name__ == "__main__":
    main()
