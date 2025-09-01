from flask import Flask, render_template, request
import joblib
import re, string

# -------------------------------
# 1. Load Model and Vectorizer
# -------------------------------
model = joblib.load("xgb_fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -------------------------------
# 2. Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    return text.strip()

# -------------------------------
# 3. Prediction Function
# -------------------------------
def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prob = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    confidence = max(prob)
    label = "🟥 Fake News" if pred == 1 else "🟩 Real News"
    return f"{label} (Confidence: {confidence:.2f})"

# -------------------------------
# 4. Flask App
# -------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        news_text = request.form.get("news")
        if news_text.strip():
            result = predict_news(news_text)
        else:
            result = "⚠️ Please enter some text!"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
