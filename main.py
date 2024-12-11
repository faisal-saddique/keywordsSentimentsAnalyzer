import streamlit as st
import pandas as pd
from datetime import datetime

from utils.api_client import APIClient
from utils.config import Config
from utils.data_processor import DataProcessor


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
                enriched_positions = DataProcessor.enrich_serp_data_with_sentiment(
                    serp_data.get("positions", [])
                )
                progress_bar.progress(60)

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
                keyword_sentiment = pd.Series(
                    comparison_results["serp_primary_emotions"]
                ).mode()[0]
                webpage_sentiment = comparison_results["webpage_primary_emotion"]
                summary = APIClient.generate_summary_and_recommendation(
                    keyword_sentiment, webpage_sentiment, webpage_content
                )

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
