import argparse
import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import trafilatura
from finvizfinance.quote import finvizfinance
from ollama import Client, ResponseError
from pydantic import BaseModel, Field, ValidationError


# ============================
# Config
# ============================

DEFAULT_MODEL = "qwen3.5"

VALID_RATINGS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
RATING_TO_SCORE = {
    "Strong Sell": -2,
    "Sell": -1,
    "Hold": 0,
    "Buy": 1,
    "Strong Buy": 2,
}
SCORE_TO_RATING = {
    -2: "Strong Sell",
    -1: "Sell",
    0: "Hold",
    1: "Buy",
    2: "Strong Buy",
}


# ============================
# Output model
# ============================

class NewsDecision(BaseModel):
    ticker: str
    rating: str
    score: int
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    time_horizon: str = "1-7 days"
    summary: str = ""
    bullish_points: List[str] = []
    bearish_points: List[str] = []
    risks: List[str] = []
    per_news_analysis: List[Dict[str, Any]] = []
    disclaimer: str = "This is not financial advice."


# ============================
# Helpers
# ============================

def clean_text(text: Optional[str], max_chars: int = 2500) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:max_chars]


def safe_col(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def extract_json_object(text: str) -> str:
    """
    Robustly extract the first JSON object from model output.
    Useful if the model accidentally adds text around the JSON.
    """
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return text

    return text[start:end + 1]


def normalize_decision(data: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    """
    Make the model output robust:
    - fills missing fields
    - converts score/rating if one is missing
    - clamps confidence
    """
    data = dict(data)

    data["ticker"] = str(data.get("ticker", ticker)).upper()

    rating = data.get("rating", "Hold")
    score = data.get("score", None)

    if isinstance(score, str):
        try:
            score = int(score)
        except Exception:
            score = None

    if rating not in VALID_RATINGS:
        rating = None

    if score not in [-2, -1, 0, 1, 2]:
        score = None

    if rating is None and score is not None:
        rating = SCORE_TO_RATING[score]

    if score is None and rating is not None:
        score = RATING_TO_SCORE[rating]

    if rating is None:
        rating = "Hold"

    if score is None:
        score = 0

    data["rating"] = rating
    data["score"] = score

    try:
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        confidence = 0.5

    data["confidence"] = max(0.0, min(1.0, confidence))

    data.setdefault("time_horizon", "1-7 days")
    data.setdefault("summary", "")
    data.setdefault("bullish_points", [])
    data.setdefault("bearish_points", [])
    data.setdefault("risks", [])
    data.setdefault("per_news_analysis", [])
    data.setdefault("disclaimer", "This is not financial advice.")

    for key in ["bullish_points", "bearish_points", "risks", "per_news_analysis"]:
        if not isinstance(data[key], list):
            data[key] = []

    return data


# ============================
# Finviz
# ============================

def fetch_finviz_news(ticker: str, max_news: int = 3) -> List[Dict[str, str]]:
    """
    Fetch latest ticker-specific Finviz news.
    """
    stock = finvizfinance(ticker.upper())
    df = stock.ticker_news()

    if df is None or df.empty:
        return []

    title_col = safe_col(df, ["Title", "title"])
    link_col = safe_col(df, ["Link", "link", "URL", "url"])
    date_col = safe_col(df, ["Date", "date", "Datetime", "datetime"])
    source_col = safe_col(df, ["Source", "source"])

    news_items = []

    for _, row in df.head(max_news).iterrows():
        headline = clean_text(row.get(title_col, "")) if title_col else ""
        link = clean_text(row.get(link_col, "")) if link_col else ""
        date = clean_text(row.get(date_col, "")) if date_col else ""
        source = clean_text(row.get(source_col, "")) if source_col else ""

        if headline:
            news_items.append(
                {
                    "date": date,
                    "source": source,
                    "headline": headline,
                    "link": link,
                    "article_text": "",
                }
            )

    return news_items


def fetch_article_text(url: str, timeout: int = 10, max_chars: int = 1500) -> str:
    """
    Try to fetch article text.
    Many finance sites block scraping, so empty text is okay.
    """
    if not url:
        return ""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        extracted = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )

        return clean_text(extracted, max_chars=max_chars)

    except Exception:
        return ""


# ============================
# Prompt
# ============================

def build_prompt(ticker: str, news_items: List[Dict[str, str]]) -> str:
    news_json = json.dumps(news_items, indent=2, ensure_ascii=False)

    return f"""
/no_think

You are a cautious financial news analyst.

Analyze the following recent public news for this stock ticker:

Ticker: {ticker.upper()}

News:
{news_json}

Your job:
Give a conservative news-based rating.

Allowed ratings:
- Strong Sell
- Sell
- Hold
- Buy
- Strong Buy

Score mapping:
- Strong Sell = -2
- Sell = -1
- Hold = 0
- Buy = 1
- Strong Buy = 2

Rules:
1. Use ONLY the provided news.
2. Do NOT invent facts.
3. If evidence is weak, stale, mixed, or unclear, choose Hold.
4. This is a short-term news sentiment score, not a full investment recommendation.
5. Return ONLY valid JSON.
6. Do not use markdown.
7. Do not include text outside JSON.

Return this exact JSON structure:

{{
  "ticker": "{ticker.upper()}",
  "rating": "Hold",
  "score": 0,
  "confidence": 0.5,
  "time_horizon": "1-7 days",
  "summary": "One short paragraph explaining the rating.",
  "bullish_points": ["point 1", "point 2"],
  "bearish_points": ["point 1", "point 2"],
  "risks": ["risk 1", "risk 2"],
  "per_news_analysis": [
    {{
      "headline": "headline text",
      "sentiment": "neutral",
      "impact_score": 0,
      "reason": "short reason"
    }}
  ],
  "disclaimer": "This is not financial advice."
}}
"""


# ============================
# Ollama
# ============================

def analyze_with_ollama(
    ticker: str,
    news_items: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    host: str = "http://localhost:11434",
) -> NewsDecision:
    client = Client(host=host)

    system_prompt = """
You are a cautious financial news analyst.
You must return only valid JSON.
Do not include markdown.
Do not include explanations outside the JSON.
"""

    user_prompt = build_prompt(ticker, news_items)

    chat_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        stream=False,
        options={
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 2048,
            "num_predict": 700,
        },
    )

    try:
        try:
            # Newer Ollama Python clients support think=False.
            response = client.chat(**chat_kwargs, think=False)
        except TypeError:
            # Older clients may not expose think=False.
            # The /no_think instruction in the prompt still helps.
            response = client.chat(**chat_kwargs)

    except ResponseError as e:
        raise RuntimeError(
            f"Ollama error: {e.error}\n\n"
            f"Try testing directly:\n"
            f"  ollama run {model} --think=false "
            f"'Return only JSON: {{\"ticker\":\"{ticker}\",\"rating\":\"Hold\",\"score\":0}}'\n"
        )

    raw = (response.message.content or "").strip()

    if not raw:
        try:
            debug_response = response.model_dump_json(indent=2)
        except Exception:
            debug_response = str(response)

        raise RuntimeError(
            "Ollama returned empty message.content.\n\n"
            "Most likely causes:\n"
            "1. Thinking mode was not disabled.\n"
            "2. The model produced reasoning but no final answer.\n"
            "3. The model hit a generation/context issue.\n\n"
            f"Full Ollama response:\n{debug_response}"
        )

    raw_json = extract_json_object(raw)

    try:
        parsed = json.loads(raw_json)
    except Exception as e:
        raise RuntimeError(
            f"Model did not return valid JSON.\n\n"
            f"Raw response:\n{raw}\n\n"
            f"JSON parse error:\n{e}"
        )

    parsed = normalize_decision(parsed, ticker)

    try:
        return NewsDecision.model_validate(parsed)
    except ValidationError as e:
        raise RuntimeError(
            f"JSON had the wrong structure.\n\n"
            f"Parsed JSON:\n{json.dumps(parsed, indent=2)}\n\n"
            f"Validation error:\n{e}"
        )


# ============================
# Printing
# ============================

def print_news(news_items: List[Dict[str, str]]) -> None:
    print("\nNews used:")
    for i, item in enumerate(news_items, start=1):
        print(f"\n[{i}] {item.get('headline', '')}")
        if item.get("date"):
            print(f"    Date: {item['date']}")
        if item.get("source"):
            print(f"    Source: {item['source']}")
        if item.get("link"):
            print(f"    Link: {item['link']}")


def print_result(result: NewsDecision) -> None:
    print("\n" + "=" * 80)
    print(f"TICKER:     {result.ticker}")
    print(f"RATING:     {result.rating}")
    print(f"SCORE:      {result.score}")
    print(f"CONFIDENCE: {result.confidence:.2f}")
    print(f"HORIZON:    {result.time_horizon}")
    print("=" * 80)

    print("\nSUMMARY")
    print(result.summary)

    print("\nBULLISH POINTS")
    if result.bullish_points:
        for p in result.bullish_points:
            print(f"- {p}")
    else:
        print("- None")

    print("\nBEARISH POINTS")
    if result.bearish_points:
        for p in result.bearish_points:
            print(f"- {p}")
    else:
        print("- None")

    print("\nRISKS")
    if result.risks:
        for r in result.risks:
            print(f"- {r}")
    else:
        print("- None")

    print("\nPER-NEWS ANALYSIS")
    if result.per_news_analysis:
        for item in result.per_news_analysis:
            print(f"\nHeadline: {item.get('headline', '')}")
            print(f"Sentiment: {item.get('sentiment', '')}")
            print(f"Impact score: {item.get('impact_score', '')}")
            print(f"Reason: {item.get('reason', '')}")
    else:
        print("- None")

    print("\nDISCLAIMER")
    print(result.disclaimer)
    print("=" * 80 + "\n")


# ============================
# Main
# ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", type=str, help="Ticker symbol, e.g. TSLA, NVDA, AAPL")
    parser.add_argument("--max-news", type=int, default=4, help="Number of latest news items to analyze, max 3")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama host")
    parser.add_argument(
        "--no-article-text",
        action="store_true",
        help="Use only Finviz headlines, dates, sources, and links",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save result to TICKER_news_rating.json",
    )

    args = parser.parse_args()

    ticker = args.ticker.upper()
    max_news = max(1, min(args.max_news, 10))

    print(f"Fetching latest Finviz news for {ticker}...")
    news_items = fetch_finviz_news(ticker, max_news=max_news)

    if not news_items:
        raise RuntimeError(f"No Finviz news found for ticker: {ticker}")

    if not args.no_article_text:
        print("Trying to fetch article text from news links...")
        for item in news_items:
            item["article_text"] = fetch_article_text(item.get("link", ""))
    else:
        print("Using headlines only.")

    print_news(news_items)

    print(f"\nAnalyzing {len(news_items)} news item(s) with Ollama model: {args.model}")
    result = analyze_with_ollama(
        ticker=ticker,
        news_items=news_items,
        model=args.model,
        host=args.host,
    )

    print_result(result)

    if args.save_json:
        out_path = f"{ticker}_news_rating.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

        print(f"Saved JSON result to: {out_path}")


if __name__ == "__main__":
    main()