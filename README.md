# 📰 Fake News Detector

A Natural Language Processing (NLP) project that classifies news headlines and articles as **REAL** or **FAKE** using text classification algorithms and TF-IDF features.

---

## 📁 Project Structure

```
fake_news_detector/
├── app.py           # Streamlit web app
├── train.py         # Model training & comparison script
├── requirements.txt # Dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
Compares Logistic Regression, Naive Bayes, Passive Aggressive Classifier, and Linear SVC — saves the best as `model.pkl`.

### 3. Launch the app
```bash
streamlit run app.py
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | ML models + TF-IDF |
| Pandas / NumPy | Data handling |
| Streamlit | Web interface |

---

## 💡 How It Works

1. Text is cleaned (lowercased, URLs & special chars removed)
2. TF-IDF with bigrams converts text to numerical features
3. Best classifier predicts REAL or FAKE with confidence score
4. Results displayed in an interactive web UI

---

## 📌 Notes

- The sample dataset is minimal for demonstration. For real-world accuracy, use:
  - [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
  - [LIAR Dataset](https://paperswithcode.com/dataset/liar)

---

*Built by Muhammed Anshad M*
