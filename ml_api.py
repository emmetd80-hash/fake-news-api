from flask import Flask, request, jsonify
from textblob import TextBlob
import joblib
import trafilatura
import requests
import re
import os

app = Flask(__name__)

# =====================
# LOAD ML MODEL
# =====================
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# =====================
# TEXT EXTRACTION
# =====================
def extract_text(url):
    print("\nExtracting URL:", url)

    try:
        downloaded = trafilatura.fetch_url(url)

        if downloaded:
            extracted = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False
            )
            if extracted:
                print("Extracted characters:", len(extracted))
                return extracted

        print("Trafilatura failed, using fallback request...")

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            extracted = trafilatura.extract(response.text)
            if extracted:
                print("Fallback extraction successful")
                return extracted

    except Exception as e:
        print("Extraction error:", e)

    return None


# =====================
# SENTIMENT ANALYSIS
# =====================
def analyse_sentiment(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity

    if score > 0.1:
        label = "Positive"
    elif score < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {"score": round(score, 3), "label": label}


# =====================
# GOOGLE FACT CHECK API
# =====================
def google_fact_check(query, api_key):
    api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    params = {"query": query, "key": api_key}

    try:
        response = requests.get(api_url, params=params, timeout=10)

        if response.status_code != 200:
            print("Google API Error:", response.text)
            return {"source_count": 0, "results": []}

        data = response.json()
        results = []

        for claim in data.get("claims", [])[:5]:
            for review in claim.get("claimReview", []):
                results.append({
                    "title": review.get("title"),
                    "url": review.get("url"),
                    "rating": review.get("textualRating"),
                    "publisher": review.get("publisher", {}).get("name")
                })

        return {"source_count": len(results), "results": results}

    except Exception as e:
        print("Google error:", e)
        return {"source_count": 0, "results": []}


# =====================
# PREDICT ROUTE
# =====================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    text = data.get("text")
    url = data.get("url")
    google_api_key = os.getenv("GOOGLE_API_KEY")  # <-- RECEIVED FROM ASP.NET

    if not google_api_key:
        return jsonify({"error": "Google API key missing"}), 400

    # Extract text from URL if provided
    if url:
        text = extract_text(url)
        if not text:
            return jsonify({"error": "Unable to extract article content from URL"}), 400

    if not text:
        return jsonify({"error": "No text supplied"}), 400

    # Machine Learning Prediction
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)

    confidence = float(max(probabilities[0])) * 100
    credibility = float(probabilities[0][1]) * 100
    label = "REAL" if prediction == 1 else "FAKE"

    # Sentiment
    sentiment = analyse_sentiment(text)

    # Claim Extraction
    sentences = re.split(r'[.!?]', text)
    keywords = ["claims", "said", "report", "according", "alleged", "fake", "hoax", "warns"]

    claim = ""
    for sentence in sentences:
        if any(word in sentence.lower() for word in keywords):
            claim = sentence.strip()
            break

    if not claim:
        claim = sentences[0].strip()

    claim = " ".join(claim.split()[:30])

    print("\nGoogle Query Sent:", claim)

    # Google Fact Check
    google_result = google_fact_check(claim, google_api_key)
    google_score = google_result["source_count"]

    # Trust Score
    external_trust = 80 if google_score > 3 else 50 if google_score > 0 else 20
    final_score = (credibility * 0.7) + (external_trust * 0.3)

    explanation = (
        f"Google Fact Check API found {google_score} related fact-check source(s)."
        if google_score > 0 else
        "No matching fact-check sources were found."
    )

    return jsonify({
        "prediction": int(prediction),
        "label": label,
        "confidence": confidence,
        "credibility": credibility,
        "sentiment": sentiment,
        "google_fact_check": google_result,
        "external_trust": external_trust,
        "final_score": final_score,
        "explanation": explanation
    })


# =====================
# START SERVER
# =====================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
