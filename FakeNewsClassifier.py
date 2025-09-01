import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

# -------------------------------
# 1. Load and Label Data
# -------------------------------
print("🔹 Loading data...")
dF = pd.read_csv("Fake.csv")
dT = pd.read_csv("True.csv")

dF["label"] = 1  # Fake
dT["label"] = 0  # Real

df = pd.concat([dF, dT], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
df = df.dropna(subset=["text", "label"])
print(f"✅ Dataset ready: {df.shape[0]} samples")

# -------------------------------
# 2. Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.strip()

print("🔹 Cleaning text...")
df["cleaned_text"] = df["text"].apply(clean_text)

# -------------------------------
# 3. Train-Test Split
# -------------------------------
print("🔹 Splitting train/test...")
X = df["cleaned_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4. TF-IDF Vectorization
# -------------------------------
print("🔹 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=10,
    max_features=10000,       # Limit to top 10k terms
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# 5. Train XGBoost Classifier
# -------------------------------
print("🔹 Training XGBoost model...")
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train_vec, y_train)
print("✅ Training complete")

# -------------------------------
# 6. Evaluation
# -------------------------------
print("🔹 Evaluating model...")
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test_vec)[:, 1]))

# -------------------------------
# 7. Save Model and Vectorizer
# -------------------------------
print("🔹 Saving model and vectorizer...")
joblib.dump(model, "xgb_fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("🎉 Saved: xgb_fake_news_model.pkl & tfidf_vectorizer.pkl")
