# 🎵 MoodTune — Sentiment-Based Music Recommendation System Using Machine Learning

> **A modern, AI-powered music recommendation web application built with Python, Streamlit, Scikit-Learn, and TF-IDF + Logistic Regression.**

---

## 🌟 Project Overview

**MoodTune** is a college machine learning project that classifies user emotional input into one of four distinct sentiments (**Happy**, **Sad**, **Relaxed**, **Motivated**) and intelligently recommends matching music tracks from a curated library.

Everything in this project runs **100% locally and free** without any paid APIs, API keys, or external cloud services.

---

## ✨ Features & Architecture

1. **🎨 Premium Dark Navy Design:**
   - Dark navy palette (`#0B0E17`, `#13182E`) with glowing purple (`#8B5CF6`) and pink (`#EC4899`) accents.
   - Glassmorphic song cards, responsive layout, and typography.
2. **💬 Natural Emotion Input & Interactive Prompt Chips:**
   - Free-form text area: *"How are you feeling today?"*
   - Clickable mood presets (e.g., *"Got a promotion! 🎉"*, *"Feeling lonely... 🌧️"*, *"Cozy Sunday chill ☕"*, *"Crushing the gym! 💪"*).
3. **🧠 Robust NLP Machine Learning Pipeline:**
   - **TF-IDF Vectorization** (unigram + bigram n-grams, sublinear term frequency scaling).
   - **Multinomial Logistic Regression** with Softmax probability outputs.
   - **Data Leakage Prevention:** Stratified train/test split (75% train, 25% test), fitting vectorizers strictly on the training partition.
4. **⚠️ Intelligent Confidence Thresholding:**
   - If the user's text is ambiguous, neutral, or has low confidence (< 34%), MoodTune displays a friendly warning rather than guessing blindly.
5. **🎧 Curated Music Recommendation:**
   - Deduplicated song cards displaying Title, Artist, Genre, Mood, Tempo (BPM), and Energy level.
   - Curated initial benchmark of 8 unique tracks (2 per sentiment).
   - Direct 1-click **Spotify** and **YouTube** search streaming links (no fake audio or broken URLs).
6. **📊 Model Performance & Evaluation Dashboard:**
   - Real-world, honest test metrics: Accuracy, Precision, Recall, and F1-Score.
   - Baseline comparison against a **Majority-Class ZeroR Baseline** (+25.0% performance gain).
   - Interactive **Plotly Confusion Matrix** and per-class performance grouped bar charts.
7. **🔍 Dataset Explorer & Label Quality Inspector:**
   - Searchable, filterable view of `dataset.csv`.
   - Realistic evaluation addressing crowdsourced label noise and concrete guidelines for dataset expansion.
8. **💡 College Viva & Theory Section:**
   - Mathematical formulations for TF-IDF and Softmax Logistic Regression rendered in LaTeX.
   - Step-by-step pipeline architecture and viva exam Q&A guide.

---

## 📁 Project Structure

```text
MoodTune/
│
├── app.py              # Main Streamlit web application & UI
├── dataset.csv         # Labeled sentiment text & song catalog
├── requirements.txt    # Python dependencies
└── README.md           # Documentation & setup guide
```

---

## 🚀 How to Run Locally

### Step 1: Open Terminal / Command Prompt
Open PowerShell, Command Prompt, or your Anaconda Prompt in this folder.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch the Application
```bash
streamlit run app.py
```
Your default web browser will automatically open:
```text
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

---

## ☁️ How to Publish to Streamlit Community Cloud (Free)

Deploying your project to the web is 100% free with Streamlit Community Cloud:

1. **Push to GitHub:**
   - Create a free account on [GitHub.com](https://github.com).
   - Create a new public repository named `moodtune-music-recommender`.
   - Upload the four project files (`app.py`, `dataset.csv`, `requirements.txt`, `README.md`).
2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
   - Click **"New app"**.
   - Select your repository (`moodtune-music-recommender`), branch (`main`), and set Main file path to `app.py`.
   - Click **"Deploy!"**.
3. **Your Live Website:**
   - Within 1–2 minutes, your website will be live with a public URL (e.g., `https://moodtune.streamlit.app`) to share with professors, evaluators, and friends!

---

## 📊 Honest Evaluation Metrics

| Metric | TF-IDF + Logistic Regression | Majority Baseline | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **50.0%** | 25.0% | **+25.0%** |
| **Macro Precision** | **52.5%** | 6.2% | **+46.3%** |
| **Macro Recall** | **50.0%** | 25.0% | **+25.0%** |
| **Macro F1-Score** | **50.0%** | 10.0% | **+40.0%** |

*Note: In accordance with academic honesty, metrics are evaluated on an unseen stratified holdout test split without fabricated accuracy or artificial manipulation.*

---

## 🛠️ Technology Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning & NLP:** [Scikit-Learn](https://scikit-learn.org/) (`TfidfVectorizer`, `LogisticRegression`, `DummyClassifier`)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
- **Visualizations:** [Plotly Express & Graph Objects](https://plotly.com/python/)
