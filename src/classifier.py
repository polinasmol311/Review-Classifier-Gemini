"""Review classification: Gemini API call + safe JSON parsing."""
import json
import os
import re
import time

SENTIMENTS = {"positive", "neutral", "negative"}
TOPICS = {"delivery", "product_quality", "price", "support", "app_bug", "design", "other"}
URGENCIES = {"low", "medium", "high"}

SYSTEM_PROMPT = f"""You classify customer reviews.
Return ONLY a JSON object, no extra text, with keys:
- "sentiment": one of {sorted(SENTIMENTS)}
- "topic": one of {sorted(TOPICS)}
- "urgency": one of {sorted(URGENCIES)} (high = needs a fast reply from the company)
- "summary": max 12 words"""

DEFAULT_MODEL = "gemini-3.8-flash"

FALLBACK = {"sentiment": "neutral", "topic": "other", "urgency": "low", "summary": ""}


def parse_response(text: str) -> dict:
    """Extract and validate the JSON; fall back to safe defaults on any problem."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return dict(FALLBACK)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return dict(FALLBACK)
    result = dict(FALLBACK)
    if data.get("sentiment") in SENTIMENTS:
        result["sentiment"] = data["sentiment"]
    if data.get("topic") in TOPICS:
        result["topic"] = data["topic"]
    if data.get("urgency") in URGENCIES:
        result["urgency"] = data["urgency"]
    result["summary"] = str(data.get("summary", ""))[:200]
    return result


def classify_review(client, review: str, model: str | None = None, retries: int = 3) -> dict:
    """Send one review to Gemini. Retries with a pause (free tier has rate limits)."""
    from google.genai import types  # imported here so dry-run/tests work without the SDK

    model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0,
    )
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model, contents=review, config=config
            )
            return parse_response(response.text or "")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))  # 2s, 4s, ...


def classify_dry_run(review: str) -> dict:
    """Keyword stub so the pipeline can be tested without an API key."""
    r = review.lower()
    neg = any(w in r for w in ("broken", "crash", "never", "fix", "bad"))
    pos = any(w in r for w in ("love", "happy", "perfect", "great"))
    return {
        "sentiment": "negative" if neg else "positive" if pos else "neutral",
        "topic": "other",
        "urgency": "high" if neg else "low",
        "summary": "(dry-run)",
    }
