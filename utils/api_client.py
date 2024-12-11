import json
from typing import Dict, Optional
from bs4 import BeautifulSoup
import html2text
import requests
import streamlit as st
from utils.config import Config

class APIClient:
    """Handles API interactions for different services."""

    @staticmethod
    def fetch_ahrefs_serp_data(keyword: str, country: str, select: str, date: Optional[str] = None, top_positions: Optional[int] = None) -> Optional[Dict]:
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
            response = requests.get(Config.AHREFS_API_URL, headers=headers, params=params)
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
            response = requests.post(Config.OPENAI_API_URL, headers=headers, json=payload)
            response.raise_for_status()

            # Parse the response
            analysis = response.json()
            if "choices" in analysis and len(analysis["choices"]) > 0:
                sentiment_data = json.loads(analysis["choices"][0]["message"]["content"])

                # Extract and ensure that we have default return values
                return {
                    "sentiment": sentiment_data.get("sentiment", "neutral"),
                    "sentiment_score": sentiment_data.get("sentiment_score", 0),
                    "primary_emotion": sentiment_data.get("primary_emotion", "none"),
                    "emotions": sentiment_data.get("emotions", []),
                    "emotional_intensity": sentiment_data.get("emotional_intensity", 0),
                    "intent": sentiment_data.get("intent", "none"),
                    "tone": sentiment_data.get("tone", "none"),
                    "key_psychological_triggers": sentiment_data.get("key_psychological_triggers", []),
                    "brief_explanation": sentiment_data.get("brief_explanation", "No explanation provided"),
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

        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
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
            soup = BeautifulSoup(html_content, "html.parser")
            body_content = soup.body

            if not body_content:
                st.warning("No body content found.")
                return None

            # Remove scripts, styles, base64 images, and links
            for tag in body_content(["script", "style", "img", "a"]):
                if tag.name == "img" and tag.get("src", "").startswith("data:image"):
                    tag.decompose()
                else:
                    tag.decompose()

            # Convert remaining HTML to markdown text
            markdown_converter = html2text.HTML2Text()
            markdown_converter.ignore_links = True
            markdown_converter.ignore_images = True
            text_content = markdown_converter.handle(str(body_content))

            # Check if the content is meaningful
            if len(text_content.strip()) < 100:  # Arbitrary threshold for meaningful content
                st.warning("Extracted content is too short to be meaningful.")
                return None

            return text_content

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching webpage content: {e}")
            return None

    @staticmethod
    def generate_summary_and_recommendation(keyword_sentiment: str, webpage_sentiment: str, webpage_content: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        }

        # Truncate webpage content to avoid exceeding token limits
        truncated_content = webpage_content[:3000]  # Adjust length as needed

        prompt = (
            "You are an expert in sentiment analysis and content optimization. Based on the following data, provide a concise summary and recommendation. "
            "The data includes the overall sentiment of a keyword, the sentiment of a webpage, and the webpage content. Use this format: "
            "'The overall sentiment for the keyword is {keyword_sentiment}, but the webpage sentiment is {webpage_sentiment}. Recommendation: {suggestion}.'\n\n"
            "Data:\n"
            f"- Keyword Sentiment: {keyword_sentiment}\n"
            f"- Webpage Sentiment: {webpage_sentiment}\n"
            f"- Webpage Content: {truncated_content}\n\n"
            "Consider the following when crafting your recommendation:\n"
            "- If the webpage sentiment is less positive than the keyword sentiment, suggest ways to make the webpage more positive.\n"
            "- If the webpage sentiment is more positive, suggest maintaining or enhancing the positive aspects.\n"
            "- Provide actionable insights for content improvement."
        )

        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 4096,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }

        try:
            response = requests.post(Config.OPENAI_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            analysis = response.json()
            if "choices" in analysis and len(analysis["choices"]) > 0:
                return analysis["choices"][0]["message"]["content"].strip()
            else:
                return "Unable to generate a summary and recommendation."
        except requests.exceptions.RequestException as e:
            return f"Error generating summary: {e}"
