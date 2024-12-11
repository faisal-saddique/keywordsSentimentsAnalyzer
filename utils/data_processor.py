# Data Processing Utilities
from typing import Dict, List

import pandas as pd

from utils.api_client import APIClient


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
        all_emotions = [emotion for item in enriched_data for emotion in item.get("emotions", [])]
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
            "top_emotions": sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "average_sentiment_score": average_sentiment_score,
            "total_positions": len(enriched_data),
        }

    def compare_with_webpage(enriched_positions: List[Dict], webpage_analysis: Dict, keyword: str) -> Dict:
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
            keyword.lower() in item.get("title", "").lower()
            for item in enriched_positions
        ) / len(enriched_positions)

        # Keyword density in webpage content
        webpage_content = webpage_analysis.get("content", "")
        if webpage_content.strip():
            webpage_keyword_density = webpage_content.lower().count(
                keyword.lower()
            ) / len(webpage_content.split())
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
