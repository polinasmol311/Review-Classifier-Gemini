# Review Classifier

Auto-classify customer reviews (sentiment, topic, urgency) with the Gemini API.
CSV in -> CSV out.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
```

## Run
```bash
python -m src.main data/sample_reviews.csv --dry-run   # no API, tests the pipeline
python -m src.main data/sample_reviews.csv             # real classification
pytest
```
Output: `data/sample_reviews_classified.csv`

## Next steps
- Batch several reviews per request to cut cost
- Compare against a hand-labelled CSV to measure accuracy
- Add a Streamlit/FastAPI front end
