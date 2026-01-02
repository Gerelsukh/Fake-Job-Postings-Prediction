import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_csv("fake_job_postings_train.csv")
test = pd.read_csv("fake_job_postings_test.csv")

# Combine text fields
def combine_text(df):
    return (
        df["company_profile"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["requirements"].fillna("") + " " +
        df["benefits"].fillna("")
    )

train["text"] = combine_text(train)
test["text"] = combine_text(test)

X = train["text"]
y = train["fraudulent"]

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)

# Logistic Regression
model = LogisticRegression(
    max_iter=300,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# Validation AUC
val_pred = model.predict_proba(X_val_vec)[:, 1]
auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", auc)

# Train on full data
X_all_vec = tfidf.fit_transform(X)
model.fit(X_all_vec, y)

# Predict test data
X_test_vec = tfidf.transform(test["text"])
test_pred = model.predict_proba(X_test_vec)[:, 1]

# Create submission
submit = pd.DataFrame({
    "job_id": test["job_id"],
    "pred": test_pred
})

submit.to_csv("submit.csv", index=False)
print("✅ submit.csv created")
