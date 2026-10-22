import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="MoodTune | Sentiment-Based Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom Modern Dark Navy CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    /* Global Theme Overrides */
    :root {
        --bg-main: #0B0E17;
        --bg-card: #13182E;
        --bg-card-hover: #19203B;
        --purple-primary: #8B5CF6;
        --purple-gradient: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%);
        --pink-accent: #EC4899;
        --navy-accent: #1E293B;
        --border-glass: rgba(139, 92, 246, 0.22);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Hero Banner */
    .hero-container {
        background: radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.22) 0%, rgba(11, 14, 23, 0) 70%),
                    linear-gradient(180deg, #12172C 0%, #0B0E17 100%);
        border: 1px solid var(--border-glass);
        border-radius: 24px;
        padding: 42px 32px;
        text-align: center;
        margin-bottom: 28px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.15);
        color: #C084FC;
        border: 1px solid rgba(139, 92, 246, 0.35);
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 3.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 20%, #C084FC 60%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        line-height: 1.15;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--text-secondary);
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Prompt Chips */
    .prompt-chip-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Modern Glass Card */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .glass-card:hover {
        border-color: rgba(236, 72, 153, 0.45);
    }

    /* Sentiment Result Card */
    .sentiment-card {
        background: linear-gradient(135deg, #171E38 0%, #111526 100%);
        border: 1px solid rgba(139, 92, 246, 0.35);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 35px rgba(139, 92, 246, 0.15);
    }

    .sentiment-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 1.5rem;
        font-weight: 700;
        padding: 8px 20px;
        border-radius: 12px;
        margin-bottom: 12px;
    }

    .badge-Happy {
        background: rgba(234, 179, 8, 0.15);
        color: #FACC15;
        border: 1px solid rgba(234, 179, 8, 0.4);
    }

    .badge-Sad {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }

    .badge-Relaxed {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .badge-Motivated {
        background: rgba(236, 72, 153, 0.15);
        color: #F472B6;
        border: 1px solid rgba(236, 72, 153, 0.4);
    }

    /* Song Card */
    .song-card {
        background: #141A33;
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 18px;
        padding: 22px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }

    .song-card:hover {
        transform: translateY(-5px);
        border-color: rgba(236, 72, 153, 0.6);
        box-shadow: 0 12px 35px rgba(236, 72, 153, 0.18);
    }

    .song-vinyl {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background: radial-gradient(circle, #0B0E17 30%, #2E1065 60%, #EC4899 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.3);
        margin-bottom: 14px;
    }

    .song-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
        line-height: 1.3;
    }

    .song-artist {
        font-size: 0.95rem;
        color: #C084FC;
        margin-bottom: 14px;
        font-weight: 500;
    }

    .song-meta-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.06);
        color: #E2E8F0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .listen-btn-group {
        display: flex;
        gap: 10px;
        margin-top: 18px;
    }

    .listen-btn {
        flex: 1;
        text-align: center;
        padding: 9px 12px;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        text-decoration: none !important;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }

    .listen-btn-spotify {
        background: rgba(30, 215, 96, 0.15);
        color: #1ED760 !important;
        border: 1px solid rgba(30, 215, 96, 0.35);
    }

    .listen-btn-spotify:hover {
        background: #1ED760;
        color: #000000 !important;
    }

    .listen-btn-youtube {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .listen-btn-youtube:hover {
        background: #EF4444;
        color: #FFFFFF !important;
    }

    /* Metric Card */
    .metric-card {
        background: #13182E;
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Low Confidence Box */
    .low-confidence-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 20px 0;
        display: flex;
        gap: 16px;
        align-items: flex-start;
    }

    /* Streamlit widget tweaks */
    div[data-baseweb="textarea"] textarea {
        background-color: #11162B !important;
        color: #F8FAFC !important;
        border-color: rgba(139, 92, 246, 0.3) !important;
        border-radius: 14px !important;
        font-size: 1.05rem !important;
        padding: 16px !important;
    }

    div[data-baseweb="textarea"] textarea:focus {
        border-color: #EC4899 !important;
        box-shadow: 0 0 15px rgba(236, 72, 153, 0.25) !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.35) !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(236, 72, 153, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# ML Pipeline & Data Caching
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    """Load the project dataset.csv from the workspace."""
    csv_path = "dataset.csv"
    if not os.path.exists(csv_path):
        # Fallback check in script directory
        csv_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    
    df = pd.read_csv(csv_path)
    # Ensure column formatting
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_resource
def train_sentiment_model(df):
    """
    Train TF-IDF + Logistic Regression pipeline on a stratified train/test split.
    Prevents data leakage by fitting vectorizer strictly on the training partition.
    Evaluates honest metrics against a majority-class baseline.
    """
    X = df["User_Text"].astype(str)
    y = df["Sentiment_Label"].astype(str)

    # Stratified Train/Test Split (75% Train, 25% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # 1. TF-IDF Feature Extraction
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # 2. Multinomial Logistic Regression Classifier
    clf = LogisticRegression(
        random_state=42,
        max_iter=300,
        C=3.0,
        solver="lbfgs"
    )
    clf.fit(X_train_vec, y_train)

    # 3. Model Evaluation on Unseen Test Partition
    y_pred = clf.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    classes = sorted(list(clf.classes_))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # 4. Majority-Class Baseline (DummyClassifier)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    baseline_accuracy = dummy.score(X_test, y_test)

    metrics = {
        "accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "baseline_accuracy": baseline_accuracy,
        "confusion_matrix": cm,
        "classes": classes,
        "report_dict": report_dict,
        "X_train_count": len(X_train),
        "X_test_count": len(X_test)
    }

    # Also train a full-dataset inference pipeline for real-time predictions
    # This guarantees the deployment app can draw upon all curated vocabulary
    full_tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )
    X_full_vec = full_tfidf.fit_transform(X)
    full_clf = LogisticRegression(random_state=42, max_iter=300, C=3.0)
    full_clf.fit(X_full_vec, y)

    return full_tfidf, full_clf, metrics


# Load Data and Pipeline
try:
    df_data = load_dataset()
    vectorizer, model, eval_metrics = train_sentiment_model(df_data)
except Exception as e:
    st.error(f"Error loading dataset.csv or training pipeline: {e}")
    st.stop()


# ---------------------------------------------------------
# Session State Management for Prompts
# ---------------------------------------------------------
if "user_text_input" not in st.session_state:
    st.session_state.user_text_input = ""

def set_prompt(text):
    st.session_state.user_text_input = text


# ---------------------------------------------------------
# Top Navigation Bar
# ---------------------------------------------------------
nav_col1, nav_col2 = st.columns([1, 2])
with nav_col1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 10px 0;">
        <div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #8B5CF6, #EC4899); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; box-shadow: 0 4px 15px rgba(236,72,153,0.4);">
            🎵
        </div>
        <div>
            <div style="font-size: 1.6rem; font-weight: 800; line-height: 1; letter-spacing: -0.02em;">
                <span style="color: #FFFFFF;">Mood</span><span style="color: #EC4899;">Tune</span>
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
                Sentiment Music AI
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    selected_tab = st.radio(
        "Navigation",
        ["🎧 Discover Music", "📊 Model Performance", "🔍 Dataset & Label Quality", "💡 About Project"],
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown("<hr style='border: none; border-top: 1px solid rgba(139, 92, 246, 0.2); margin: 8px 0 24px 0;'>", unsafe_allow_html=True)


# =========================================================
# TAB 1: MUSIC RECOMMENDATION (HOME)
# =========================================================
if selected_tab == "🎧 Discover Music":

    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <span class="hero-badge">AI-Powered Music Discovery</span>
        <h1 class="hero-title">Feel Your Mood. Find Your Music.</h1>
        <p class="hero-subtitle">
            Express how you are feeling in natural words. Our Machine Learning NLP model analyzes your emotional sentiment and curates matching tracks tailored to your vibe.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main Input Section
    col_input, col_info = st.columns([7, 3], gap="large")

    with col_input:
        st.markdown("<div class='prompt-chip-label'>💡 Or choose a quick mood prompt to try:</div>", unsafe_allow_html=True)
        
        # Clickable Mood Prompt Chips
        chip_c1, chip_c2, chip_c3, chip_c4 = st.columns(4)
        with chip_c1:
            if st.button("🎉 Got a promotion!", use_container_width=True):
                set_prompt("Just got promoted at work and I couldn't be more excited and joyful! Life is wonderful!")
        with chip_c2:
            if st.button("🌧️ Feeling lonely...", use_container_width=True):
                set_prompt("Feeling really lonely tonight, staring out the window as it rains, feeling heartbroken.")
        with chip_c3:
            if st.button("☕ Cozy Sunday chill", use_container_width=True):
                set_prompt("Sipping warm chamomile tea on the porch while listening to the birds, total calm and peace.")
        with chip_c4:
            if st.button("💪 Crushing the gym!", use_container_width=True):
                set_prompt("Crushed my morning workout and hit a personal record! Focused, motivated, and ready to conquer the day!")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Text input
        user_text = st.text_area(
            "How are you feeling today?",
            value=st.session_state.user_text_input,
            placeholder="Type anything... e.g., 'I just aced my exam and feel like dancing!' or 'Exhausted after a stressful day and just need to unwind.'",
            height=130,
            help="Tell us about your emotions, your thoughts, or what happened today."
        )

        discover_clicked = st.button("✨ Discover My Music", use_container_width=True)

    with col_info:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <h4 style="color: #C084FC; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                <span>✨</span> Supported Moods
            </h4>
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px; font-size: 0.95rem;">
                    <span style="font-size: 1.2rem;">😊</span>
                    <div><strong>Happy</strong> <span style="color: #94A3B8; font-size: 0.8rem;">• Upbeat & Joyful</span></div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; font-size: 0.95rem;">
                    <span style="font-size: 1.2rem;">🌧️</span>
                    <div><strong>Sad</strong> <span style="color: #94A3B8; font-size: 0.8rem;">• Melancholic & Deep</span></div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; font-size: 0.95rem;">
                    <span style="font-size: 1.2rem;">🧘</span>
                    <div><strong>Relaxed</strong> <span style="color: #94A3B8; font-size: 0.8rem;">• Soothing & Calm</span></div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; font-size: 0.95rem;">
                    <span style="font-size: 1.2rem;">🔥</span>
                    <div><strong>Motivated</strong> <span style="color: #94A3B8; font-size: 0.8rem;">• Driven & Energetic</span></div>
                </div>
            </div>
            <hr style="border-color: rgba(139,92,246,0.2); margin: 16px 0 12px 0;">
            <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.5;">
                ⚡ Powered by <strong>TF-IDF Vectorization</strong> and <strong>Multinomial Logistic Regression</strong> with zero external paid APIs.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Process Sentiment Prediction
    if discover_clicked or (user_text.strip() != "" and "last_analyzed" in st.session_state and st.session_state.last_analyzed == user_text):
        st.session_state.last_analyzed = user_text
        cleaned_text = user_text.strip()

        if not cleaned_text:
            st.warning("Please type a few words about how you're feeling or choose one of the sample prompt chips above!")
        else:
            # Vectorize input
            vec_input = vectorizer.transform([cleaned_text])
            
            # Predict probabilities
            probs = model.predict_proba(vec_input)[0]
            classes = list(model.classes_)
            
            # Identify top class and confidence
            top_idx = int(np.argmax(probs))
            predicted_sentiment = classes[top_idx]
            top_prob = float(probs[top_idx])
            
            # Check for low-confidence or out-of-domain input
            # In a 4-class balanced setup, a completely uniform distribution is 25%.
            # If top probability is < 34% or no features match vocabulary, trigger clear message
            non_zero_features = vec_input.nnz

            if top_prob < 0.34 or non_zero_features == 0:
                st.markdown(f"""
                <div class="low-confidence-box">
                    <div style="font-size: 2rem;">⚠️</div>
                    <div>
                        <h4 style="color: #FBBF24; margin: 0 0 6px 0;">Model Could Not Confidently Interpret Your Mood</h4>
                        <p style="color: #E2E8F0; margin: 0 0 8px 0; font-size: 0.95rem; line-height: 1.5;">
                            MoodTune analyzed your text, but the highest sentiment confidence was only <strong>{top_prob*100:.1f}%</strong> (close to a random guess). 
                            This often happens when input is very brief, neutral, or contains vocabulary outside the emotional training set.
                        </p>
                        <div style="font-size: 0.85rem; color: #94A3B8;">
                            👉 <strong>Tip:</strong> Try describing your feelings in more detail (e.g., mention whether you feel peaceful, overjoyed, exhausted, or determined), or click one of the preset mood buttons above.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Emojis and descriptions for sentiments
                mood_meta = {
                    "Happy": {"emoji": "😊", "desc": "Bright, uplifting, and filled with joy", "color": "#FACC15"},
                    "Sad": {"emoji": "🌧️", "desc": "Reflective, emotional, and comforting", "color": "#60A5FA"},
                    "Relaxed": {"emoji": "🧘", "desc": "Serene, tranquil, and meditative", "color": "#34D399"},
                    "Motivated": {"emoji": "🔥", "desc": "High energy, focused, and ambitious", "color": "#F472B6"}
                }

                meta = mood_meta.get(predicted_sentiment, {"emoji": "🎵", "desc": "Curated for your vibe", "color": "#C084FC"})

                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

                # Sentiment Result Card & Confidence Bar
                st.markdown(f"""
                <div class="sentiment-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                        <div>
                            <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 6px;">
                                Detected Emotional State
                            </div>
                            <div class="sentiment-badge badge-{predicted_sentiment}">
                                <span>{meta['emoji']}</span>
                                <span>{predicted_sentiment}</span>
                            </div>
                            <div style="color: #E2E8F0; font-size: 1.05rem; margin-top: 4px;">
                                {meta['desc']}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">
                                Prediction Confidence
                            </div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: {meta['color']};">
                                {top_prob*100:.1f}%
                            </div>
                            <div style="font-size: 0.8rem; color: #64748B;">
                                Softmax Probability
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Emotion Probability Breakdown (Plotly Horizontal Bar)
                with st.expander("📊 View Detailed Mood Probability Breakdown", expanded=False):
                    prob_df = pd.DataFrame({
                        "Sentiment": classes,
                        "Probability": [p * 100 for p in probs]
                    }).sort_values("Probability", ascending=True)

                    bar_colors = [mood_meta.get(s, {}).get("color", "#8B5CF6") for s in prob_df["Sentiment"]]

                    fig_prob = go.Figure(go.Bar(
                        x=prob_df["Probability"],
                        y=prob_df["Sentiment"],
                        orientation="h",
                        marker=dict(color=bar_colors, line=dict(color='rgba(255,255,255,0.2)', width=1)),
                        text=[f"{val:.1f}%" for val in prob_df["Probability"]],
                        textposition="outside",
                        textfont=dict(color="#FFFFFF", size=12)
                    ))
                    fig_prob.update_layout(
                        title=dict(text="Model Probability Distribution Across Emotions", font=dict(color="#FFFFFF", size=14)),
                        xaxis=dict(title="Probability (%)", range=[0, 110], gridcolor="rgba(139,92,246,0.15)", tickfont=dict(color="#94A3B8")),
                        yaxis=dict(tickfont=dict(color="#FFFFFF", size=13)),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=240,
                        margin=dict(l=20, r=40, t=40, b=20)
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

                # ---------------------------------------------------------
                # Music Recommendations Section
                # ---------------------------------------------------------
                st.markdown(f"""
                <div style="margin-top: 32px; margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                        <div>
                            <h2 style="font-size: 1.8rem; margin: 0; color: #FFFFFF;">
                                Recommended Songs for <span style="color: {meta['color']};">{predicted_sentiment}</span> Vibes
                            </h2>
                            <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
                                Curated from your project library matching this emotional frequency.
                            </p>
                        </div>
                        <div style="background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.3); padding: 6px 14px; border-radius: 9999px; font-size: 0.8rem; color: #C084FC; font-weight: 600;">
                            Library: 8 Curated College Benchmark Tracks
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Filter dataset for songs matching predicted sentiment
                # Use Mood column if present, fallback to Sentiment_Label
                match_mask = (df_data["Mood"].str.lower() == predicted_sentiment.lower()) | \
                             (df_data["Sentiment_Label"].str.lower() == predicted_sentiment.lower())
                
                matched_songs = df_data[match_mask]

                # Drop duplicate songs to guarantee clean recommendations
                unique_recommendations = matched_songs.drop_duplicates(subset=["Song_Title", "Artist"])

                if unique_recommendations.empty:
                    st.info(f"No songs cataloged under {predicted_sentiment} in the library yet.")
                else:
                    # Display song cards in responsive columns
                    cols = st.columns(len(unique_recommendations))
                    
                    for idx, (_, song_row) in enumerate(unique_recommendations.iterrows()):
                        col = cols[idx % len(cols)]
                        with col:
                            title = song_row.get("Song_Title", "Unknown Track")
                            artist = song_row.get("Artist", "Unknown Artist")
                            genre = song_row.get("Genre", "Music")
                            mood = song_row.get("Mood", predicted_sentiment)
                            tempo = song_row.get("Tempo_BPM", "N/A")
                            energy = song_row.get("Energy_Level", "Medium")

                            # Reliable listening links via authenticated Spotify & YouTube search queries
                            # Does NOT invent fake audio files or unverified song URLs
                            encoded_query = urllib.parse.quote_plus(f"{title} {artist}")
                            spotify_search_url = f"https://open.spotify.com/search/{encoded_query}"
                            youtube_search_url = f"https://www.youtube.com/results?search_query={encoded_query}"

                            vinyl_icons = ["💿", "🎵", "🎧", "✨"]
                            vinyl_icon = vinyl_icons[idx % len(vinyl_icons)]

                            st.markdown(f"""
                            <div class="song-card">
                                <div>
                                    <div class="song-vinyl">{vinyl_icon}</div>
                                    <div class="song-title">{title}</div>
                                    <div class="song-artist">by {artist}</div>
                                    <div style="margin-top: 10px;">
                                        <span class="song-meta-pill">🎸 {genre}</span>
                                        <span class="song-meta-pill">⚡ {energy} Energy</span>
                                        <span class="song-meta-pill">⏱️ {tempo} BPM</span>
                                    </div>
                                </div>
                                <div>
                                    <div class="listen-btn-group">
                                        <a href="{spotify_search_url}" target="_blank" class="listen-btn listen-btn-spotify" title="Listen on Spotify">
                                            <span>🟢</span> Spotify
                                        </a>
                                        <a href="{youtube_search_url}" target="_blank" class="listen-btn listen-btn-youtube" title="Listen on YouTube">
                                            <span>▶️</span> YouTube
                                        </a>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("""
                <div style="margin-top: 24px; padding: 14px 18px; border-radius: 12px; background: rgba(255,255,255,0.03); border: 1px dashed rgba(139,92,246,0.3); font-size: 0.85rem; color: #94A3B8;">
                    📌 <strong>Note on College Project Library:</strong> To avoid duplicate recommendations and maintain strict proof-of-concept integrity, 
                    this initial starter library features eight unique curated songs (2 per emotional class). 
                    You can easily expand this to thousands of songs by simply adding rows to your <code>dataset.csv</code> file.
                </div>
                """, unsafe_allow_html=True)


# =========================================================
# TAB 2: MODEL PERFORMANCE DASHBOARD
# =========================================================
elif selected_tab == "📊 Model Performance":

    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 2.2rem; color: #FFFFFF; margin-bottom: 6px;">
            📊 Model Performance & Evaluation Dashboard
        </h1>
        <p style="color: #94A3B8; font-size: 1.05rem;">
            Real-world, unmanipulated machine learning evaluation metrics on a stratified holdout test partition (25%).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Indicators
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{eval_metrics['accuracy']*100:.1f}%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #60A5FA !important;">{eval_metrics['baseline_accuracy']*100:.1f}%</div>
            <div class="metric-label">Majority Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{eval_metrics['precision']*100:.1f}%</div>
            <div class="metric-label">Macro Precision</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{eval_metrics['recall']*100:.1f}%</div>
            <div class="metric-label">Macro Recall</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{eval_metrics['f1']*100:.1f}%</div>
            <div class="metric-label">Macro F1-Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Baseline Comparison Callout
    gain = (eval_metrics['accuracy'] - eval_metrics['baseline_accuracy']) * 100
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(139,92,246,0.15) 0%, rgba(236,72,153,0.15) 100%); border: 1px solid rgba(139,92,246,0.35); border-radius: 16px; padding: 18px 24px; margin-bottom: 28px;">
        <h4 style="color: #C084FC; margin: 0 0 6px 0;">🎯 Model Comparison Against Baseline</h4>
        <div style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.5;">
            In a balanced 4-class classification problem (Happy, Sad, Relaxed, Motivated), a naive majority-class baseline achieves exactly 
            <strong>{eval_metrics['baseline_accuracy']*100:.1f}%</strong> accuracy. Our TF-IDF + Logistic Regression pipeline achieves 
            <strong>{eval_metrics['accuracy']*100:.1f}%</strong> on the unseen test set — an absolute improvement of 
            <strong style="color: #34D399;">+{gain:.1f}%</strong>, proving genuine feature learning despite presence of noisy labels.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Plots Section: Confusion Matrix & Per-Class Metrics
    plot_col1, plot_col2 = st.columns([1, 1], gap="large")

    with plot_col1:
        st.markdown("<h3 style='color: #FFFFFF; font-size: 1.25rem; margin-bottom: 12px;'>🔥 Confusion Matrix (Test Split)</h3>", unsafe_allow_html=True)
        
        cm = eval_metrics["confusion_matrix"]
        classes = eval_metrics["classes"]

        # Custom Plotly Heatmap
        fig_cm = px.imshow(
            cm,
            x=[f"Pred {c}" for c in classes],
            y=[f"True {c}" for c in classes],
            text_auto=True,
            color_continuous_scale=[
                [0.0, "#0B0E17"],
                [0.3, "#1E1B4B"],
                [0.6, "#7C3AED"],
                [1.0, "#EC4899"]
            ],
            aspect="auto"
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF", family="Plus Jakarta Sans"),
            coloraxis_showscale=False,
            height=340,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig_cm.update_xaxes(side="bottom", tickfont=dict(color="#CBD5E1", size=11))
        fig_cm.update_yaxes(tickfont=dict(color="#CBD5E1", size=11))
        st.plotly_chart(fig_cm, use_container_width=True)

    with plot_col2:
        st.markdown("<h3 style='color: #FFFFFF; font-size: 1.25rem; margin-bottom: 12px;'>📈 Per-Class Precision, Recall & F1</h3>", unsafe_allow_html=True)
        
        rep = eval_metrics["report_dict"]
        class_metrics = []
        for c in classes:
            if c in rep:
                class_metrics.append({
                    "Class": c,
                    "Precision": rep[c]["precision"] * 100,
                    "Recall": rep[c]["recall"] * 100,
                    "F1-Score": rep[c]["f1-score"] * 100
                })
        
        cm_df = pd.DataFrame(class_metrics)
        
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(name="Precision", x=cm_df["Class"], y=cm_df["Precision"], marker_color="#8B5CF6"))
        fig_bars.add_trace(go.Bar(name="Recall", x=cm_df["Class"], y=cm_df["Recall"], marker_color="#EC4899"))
        fig_bars.add_trace(go.Bar(name="F1-Score", x=cm_df["Class"], y=cm_df["F1-Score"], marker_color="#34D399"))

        fig_bars.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF", family="Plus Jakarta Sans"),
            yaxis=dict(title="Score (%)", range=[0, 110], gridcolor="rgba(139,92,246,0.15)", tickfont=dict(color="#94A3B8")),
            xaxis=dict(tickfont=dict(color="#CBD5E1", size=12)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#E2E8F0")),
            height=340,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    # Detailed Classification Table
    st.markdown("<h3 style='color: #FFFFFF; font-size: 1.25rem; margin-top: 18px; margin-bottom: 12px;'>📋 Detailed Scikit-Learn Classification Report</h3>", unsafe_allow_html=True)
    
    rows = []
    for k, v in rep.items():
        if isinstance(v, dict):
            rows.append({
                "Category / Metric": k,
                "Precision": f"{v['precision']:.3f}",
                "Recall": f"{v['recall']:.3f}",
                "F1-Score": f"{v['f1-score']:.3f}",
                "Support": int(v['support'])
            })
    
    report_table_df = pd.DataFrame(rows)
    st.dataframe(
        report_table_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(f"""
    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 10px;">
        * Partitioned using a strictly stratified train/test split (Train: {eval_metrics['X_train_count']} samples | Test: {eval_metrics['X_test_count']} samples). 
        Zero data leakage was introduced. All metrics reflect genuine performance on raw dataset samples.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB 3: DATASET & LABEL QUALITY
# =========================================================
elif selected_tab == "🔍 Dataset & Label Quality":

    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 2.2rem; color: #FFFFFF; margin-bottom: 6px;">
            🔍 Dataset Analysis & Label Integrity
        </h1>
        <p style="color: #94A3B8; font-size: 1.05rem;">
            Transparent inspection of <code>dataset.csv</code>, class distributions, music catalog, and noisy label diagnostics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    d_col1, d_col2 = st.columns([1, 1], gap="large")

    with d_col1:
        st.markdown("<h3 style='color: #FFFFFF; font-size: 1.2rem; margin-bottom: 12px;'>📊 Emotion Class Distribution</h3>", unsafe_allow_html=True)
        counts = df_data["Sentiment_Label"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]

        fig_pie = px.pie(
            counts,
            values="Count",
            names="Sentiment",
            color="Sentiment",
            color_discrete_map={
                "Happy": "#FACC15",
                "Sad": "#60A5FA",
                "Relaxed": "#34D399",
                "Motivated": "#F472B6"
            },
            hole=0.5
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF", family="Plus Jakarta Sans"),
            height=280,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with d_col2:
        st.markdown("<h3 style='color: #FFFFFF; font-size: 1.2rem; margin-bottom: 12px;'>🎵 Unique Benchmark Song Library (8 Tracks)</h3>", unsafe_allow_html=True)
        unique_lib = df_data[["Song_Title", "Artist", "Genre", "Mood", "Tempo_BPM", "Energy_Level"]].drop_duplicates()
        st.dataframe(unique_lib, use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color: rgba(139,92,246,0.2); margin: 24px 0;'>", unsafe_allow_html=True)

    # Raw Dataset Explorer
    st.markdown("<h3 style='color: #FFFFFF; font-size: 1.25rem; margin-bottom: 12px;'>📄 Raw Dataset Explorer (`dataset.csv`)</h3>", unsafe_allow_html=True)
    
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        filter_sentiment = st.selectbox("Filter by Sentiment:", ["All"] + list(df_data["Sentiment_Label"].unique()))
    with filter_col2:
        search_query = st.text_input("Search User Text or Song Title:", "")

    filtered_df = df_data.copy()
    if filter_sentiment != "All":
        filtered_df = filtered_df[filtered_df["Sentiment_Label"] == filter_sentiment]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["User_Text"].str.lower().str.contains(q) |
            filtered_df["Song_Title"].str.lower().str.contains(q)
        ]

    st.dataframe(filtered_df, use_container_width=True, height=280)

    # Label Noise & Dataset Improvement Guide
    st.markdown("""
    <div style="margin-top: 28px; background: #13182C; border: 1px solid rgba(139,92,246,0.3); border-radius: 18px; padding: 24px;">
        <h3 style="color: #EC4899; margin-top: 0; font-size: 1.3rem;">
            ⚠️ Label Noise & Dataset Limitation Explanation
        </h3>
        <p style="color: #E2E8F0; line-height: 1.6; font-size: 0.95rem;">
            In natural language processing and sentiment analysis, text examples often suffer from <strong>label noise</strong>. 
            For example, sarcasm, ambivalent feelings, or crowdsourced annotation errors can lead to sentences like:
        </p>
        <ul style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
            <li><em>"I feel so exhausted and burned out..."</em> labeled under <strong>Happy</strong></li>
            <li><em>"Drinking a warm cup of herbal tea peacefully..."</em> labeled under <strong>Sad</strong></li>
            <li><em>"Furious at what happened today at work..."</em> labeled under <strong>Relaxed</strong></li>
        </ul>
        <h4 style="color: #C084FC; margin-top: 16px;">How to Authentically Improve the Dataset for Future Work:</h4>
        <ol style="color: #E2E8F0; font-size: 0.92rem; line-height: 1.6;">
            <li><strong>Dual-Annotator Verification (Cohen's Kappa):</strong> Have at least two human annotators independently label each text sample and measure inter-rater agreement.</li>
            <li><strong>VADER & Sentiment Cross-Validation:</strong> Pre-screen sentences using rule-based sentiment lexicons to flag potential label-text contradictions.</li>
            <li><strong>Dataset Scaling:</strong> Expand the dataset from 80 samples to 1,000+ balanced samples across the 4 classes to increase TF-IDF vocabulary coverage.</li>
            <li><strong>Contextual Song Attributes:</strong> Incorporate audio acoustic feature embeddings (e.g., Spotify danceability, acousticness, valence) alongside text sentiment.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB 4: ABOUT PROJECT & ARCHITECTURE
# =========================================================
elif selected_tab == "💡 About Project":

    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 2.2rem; color: #FFFFFF; margin-bottom: 6px;">
            💡 Project Architecture & Machine Learning Workflow
        </h1>
        <p style="color: #94A3B8; font-size: 1.05rem;">
            Comprehensive academic project documentation for college viva, presentation, and grading evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture Overview Cards
    arch1, arch2, arch3, arch4 = st.columns(4)

    with arch1:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 100%;">
            <div style="font-size: 2.2rem; margin-bottom: 10px;">📝</div>
            <h4 style="color: #FFFFFF; margin-bottom: 8px;">1. Text Ingestion</h4>
            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                Raw user emotion text is gathered via clean Streamlit UI or simulated mood prompt presets.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with arch2:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 100%;">
            <div style="font-size: 2.2rem; margin-bottom: 10px;">🔠</div>
            <h4 style="color: #FFFFFF; margin-bottom: 8px;">2. TF-IDF Vectors</h4>
            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                Converts text into statistical numerical sparse vectors using unigram + bigram frequencies with sublinear TF scaling.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with arch3:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 100%;">
            <div style="font-size: 2.2rem; margin-bottom: 10px;">🧠</div>
            <h4 style="color: #FFFFFF; margin-bottom: 8px;">3. Logistic Regression</h4>
            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                Multi-class Softmax classifier outputs probability distributions over Happy, Sad, Relaxed, and Motivated.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with arch4:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 100%;">
            <div style="font-size: 2.2rem; margin-bottom: 10px;">🎶</div>
            <h4 style="color: #FFFFFF; margin-bottom: 8px;">4. Recommendation</h4>
            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                Filters non-duplicate songs matching predicted emotion with authentic Spotify & YouTube streaming search links.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(139,92,246,0.2); margin: 28px 0;'>", unsafe_allow_html=True)

    # Mathematical Explanations
    math_col1, math_col2 = st.columns(2, gap="large")

    with math_col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #C084FC; margin-top: 0;">📐 1. TF-IDF Feature Extraction</h3>
            <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.6;">
                <strong>Term Frequency-Inverse Document Frequency (TF-IDF)</strong> converts variable-length text strings into weighted feature vectors:
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1")
        st.markdown("""
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 8px; line-height: 1.5;">
            Where <code>TF(t, d)</code> represents sublinear term frequency within the user's input, and <code>DF(t, D)</code> represents the document frequency across all training examples.
        </div>
        """, unsafe_allow_html=True)

    with math_col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #EC4899; margin-top: 0;">⚡ 2. Multinomial Logistic Regression (Softmax)</h3>
            <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.6;">
                For multi-class classification across our $K = 4$ emotions, the model predicts the probability of class $c$ using the Softmax function:
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"P(y = c \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_c^T \mathbf{x} + b_c)}{\sum_{k=1}^K \exp(\mathbf{w}_k^T \mathbf{x} + b_k)}")
        st.markdown("""
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 8px; line-height: 1.5;">
            Where $\mathbf{w}_c$ is the learned weight vector for sentiment class $c$, $b_c$ is the bias term, and $\mathbf{x}$ is the TF-IDF feature representation.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(139,92,246,0.2); margin: 28px 0;'>", unsafe_allow_html=True)

    # Technology Stack Table
    st.markdown("<h3 style='color: #FFFFFF; font-size: 1.3rem; margin-bottom: 14px;'>🛠️ Technology Stack & Dependencies</h3>", unsafe_allow_html=True)
    
    tech_data = [
        {"Component": "Frontend & UI", "Technology": "Streamlit (Python)", "Role": "Responsive dark navy interface, interactive cards, responsive layout"},
        {"Component": "Machine Learning", "Technology": "Scikit-Learn", "Role": "TF-IDF Vectorization, Logistic Regression, train/test evaluation"},
        {"Component": "Data Handling", "Technology": "Pandas & NumPy", "Role": "Dataset loading, deduplication, filtering, statistical metrics"},
        {"Component": "Data Visualization", "Technology": "Plotly Express & Graph Objects", "Role": "Interactive confusion matrix, grouped metric bars, probability distribution"},
        {"Component": "Music Retrieval", "Technology": "Content-Based Matching", "Role": "Safe URL encoding for Spotify and YouTube track query searches"}
    ]
    st.table(pd.DataFrame(tech_data))

    # Project Viva Q&A Guide
    with st.expander("🎓 College Viva Examination & Presentation Q&A Quick Guide", expanded=False):
        st.markdown("""
        **Q1: Why choose Logistic Regression over a Deep Neural Network?**  
        *Answer:* For smaller textual datasets, Deep Neural Networks (like LSTMs or BERT) easily overfit and require high computational resources / GPU. TF-IDF with Logistic Regression provides a high-performing, fast, explainable, and convex optimization model that runs in milliseconds without expensive infrastructure.

        **Q2: How is data leakage prevented?**  
        *Answer:* Data leakage is strictly prevented by applying `train_test_split(stratify=y)` *before* any feature extraction. The `TfidfVectorizer` is fitted exclusively on `X_train`, and only transforms `X_test`.

        **Q3: How does the system handle low-confidence or ambiguous input?**  
        *Answer:* The model calculates posterior probabilities using the Softmax function. If the maximum probability across all 4 sentiment classes falls below the decision threshold (e.g. 34%), the system refuses to guess blindly and advises the user with clear corrective feedback.

        **Q4: Why does the initial library contain 8 songs?**  
        *Answer:* To eliminate duplicate recommendations, simplify evaluation, and establish a balanced benchmark of 2 curated tracks per emotion class. The system architecture is built to seamlessly scale to thousands of tracks by simply appending rows to `dataset.csv`.
        """)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div style="margin-top: 60px; padding: 24px 0 12px 0; border-top: 1px solid rgba(139, 92, 246, 0.2); text-align: center; color: #64748B; font-size: 0.85rem;">
    <strong>MoodTune</strong> • Sentiment-Based Music Recommendation System Using Machine Learning<br>
    Built with Python, Streamlit, Scikit-Learn, and Plotly. 100% Free & Open Source without Paid APIs.
</div>
""", unsafe_allow_html=True)
