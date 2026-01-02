# Fake-Job-Postings-Prediction

## 📌 Project Overview
This project focuses on detecting **fraudulent job postings** using **Natural Language Processing (NLP)** and **machine learning classification models**.  
By analyzing textual information from job advertisements, the model learns to distinguish between **real** and **fake** job postings.

The project covers the full machine learning pipeline, including data preprocessing, feature extraction, model training, and evaluation.

---

## 📊 Dataset
The dataset contains job posting information such as:
- Company profile
- Job description
- Requirements
- Benefits
- Fraud label (real or fake)

Multiple text fields are combined to form a single textual input for the model.

---

## 🛠 Methodology

### 1. Text Preprocessing
- Missing values are handled
- Relevant text fields are merged into a single feature
- English stop words are removed

### 2. Feature Extraction
- **TF-IDF (Term Frequency–Inverse Document Frequency)** is used
- Unigrams and bigrams are included to capture meaningful phrases
- Feature size is limited to control model complexity

### 3. Model
- **Logistic Regression** classifier
- Trained on TF-IDF features

### 4. Evaluation
- Train / validation split
- Performance evaluated using **ROC-AUC score**

---

## 📈 Results
The model demonstrates strong performance in distinguishing fraudulent job postings by leveraging textual patterns and discriminative keywords commonly found in fake advertisements.

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Fake-Job-Postings-Prediction.git
cd Fake-Job-Postings-Prediction

pip install -r requirements.txt

python src/train.py
