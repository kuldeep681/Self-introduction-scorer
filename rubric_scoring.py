import streamlit as st
import nltk
nltk.download('vader_lexicon')
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
import language_tool_python

# ================= HARD-CODED RUBRIC RULES =================

SALUTATION_KEYWORDS = {
    "Excellent": ["i am excited to introduce", "feeling great"],
    "Good": ["good morning", "good afternoon", "good evening", "good day", "hello everyone"],
    "Normal": ["hi", "hello"],
}
SALUTATION_SCORES = {"Excellent": 5, "Good": 4, "Normal": 2, "None": 0}

MANDATORY_KEYWORDS = ["name", "age", "school", "class", "family", "hobbies", "interest", "free time"]
MANDATORY_POINT_PER = 4 

GOOD_TO_HAVE_KEYWORDS = [
    "about family",
    "i am from",
    "parents are from",
    "ambition",
    "goal",
    "dream",
    "interesting",
    "fun fact",
    "unique",
    "strength",
    "achievement",
]
GOOD_TO_HAVE_POINT_PER = 2

FILLER_WORDS = [
    "um", "uh", "like", "you know", "so", "actually", "basically",
    "right", "i mean", "well", "kinda", "sort of", "okay", "hmm", "ah"
]

# =================== SCORING FUNCTIONS ===================

def score_salutation(text):
    text_lower = text.lower()
    for level, keywords in SALUTATION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return SALUTATION_SCORES[level]
    return SALUTATION_SCORES["None"]

def score_keywords(text):
    text_lower = text.lower()
    mandatory_found = sum(1 for kw in MANDATORY_KEYWORDS if kw in text_lower)
    good_found = sum(1 for kw in GOOD_TO_HAVE_KEYWORDS if kw in text_lower)

    mandatory_score = min(mandatory_found * MANDATORY_POINT_PER, 20)
    good_score = min(good_found * GOOD_TO_HAVE_POINT_PER, 10)

    return mandatory_score + good_score, mandatory_score, good_score

def check_flow(text):
    text_lower = text.lower()
    flow_order = [
        ("Salutation", ["hi", "hello", "good morning", "hello everyone", "i am excited to introduce", "feeling great"]),
        ("Name", ["name"]), 
        ("Mandatory details", MANDATORY_KEYWORDS),
        ("Optional details", GOOD_TO_HAVE_KEYWORDS),
        ("Closing", ["thank you", "thanks", "thank"]),
    ]

    positions = []
    for label, keywords in flow_order:
        pos = -1
        for kw in keywords:
            found_pos = text_lower.find(kw)
            if found_pos != -1:
                if pos == -1 or found_pos < pos:
                    pos = found_pos
        positions.append(pos if pos != -1 else float('inf'))

    # Checking if positions are strictly increasing
    filtered_positions = [p for p in positions if p != float('inf')]
    if filtered_positions == sorted(filtered_positions):
        return 5
    else:
        return 0

def score_speech_rate(word_count, duration_sec):
    wpm = word_count / (duration_sec / 60) if duration_sec > 0 else 0
    if wpm > 161:
        score = 2
    elif 141 <= wpm <= 160:
        score = 6
    elif 111 <= wpm <= 140:
        score = 10
    elif 81 <= wpm <= 110:
        score = 6
    elif wpm < 80:
        score = 2
    else:
        score = 0
    return score, wpm

def score_grammar(text, word_count):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    errors = len(matches)
    errors_per_100_words = errors / (word_count / 100) if word_count > 0 else 100

    grammar_score_val = 1 - min(errors_per_100_words / 10, 1)
    if grammar_score_val > 0.9:
        return 10, errors_per_100_words
    elif 0.7 <= grammar_score_val <= 0.89:
        return 8, errors_per_100_words
    elif 0.5 <= grammar_score_val <= 0.69:
        return 6, errors_per_100_words
    elif 0.3 <= grammar_score_val <= 0.49:
        return 4, errors_per_100_words
    else:
        return 2, errors_per_100_words

def score_vocabulary(text, word_count):
    words = text.lower().split()
    unique_words = set(words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0

    if ttr >= 0.9:
        return 10, ttr
    elif 0.7 <= ttr < 0.9:
        return 8, ttr
    elif 0.5 <= ttr < 0.7:
        return 6, ttr
    elif 0.3 <= ttr < 0.5:
        return 4, ttr
    else:
        return 2, ttr

def score_clarity(text, word_count):
    text_lower = text.lower()
    filler_count = sum(text_lower.count(fw) for fw in FILLER_WORDS)
    filler_rate = (filler_count / word_count) * 100 if word_count > 0 else 100

    if 0 <= filler_rate <= 3:
        score = 15
    elif 4 <= filler_rate <= 6:
        score = 12
    elif 7 <= filler_rate <= 9:
        score = 9
    elif 10 <= filler_rate <= 12:
        score = 6
    else:
        score = 3
    return score, filler_rate

def score_engagement(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    positive = sentiment_scores['pos']

    if positive >= 0.9:
        score = 15
    elif 0.7 <= positive < 0.9:
        score = 12
    elif 0.5 <= positive < 0.7:
        score = 9
    elif 0.3 <= positive < 0.5:
        score = 6
    else:
        score = 3
    return score, positive

# =================== STREAMLIT DEPLOYMENT ===================

def main():
    st.title("Self Introduction Scorer")

    introduction = st.text_area("Paste your self introduction here:", height=250)
    duration_sec = st.number_input("Enter duration (in seconds):", min_value=1.0, step=0.1)

    if st.button("Calculate Scores"):
        if not introduction.strip():
            st.warning("Please enter your introduction before scoring.")
            return

        word_count = len(introduction.split())
        sentence_count = len(sent_tokenize(introduction, language='english'))

        st.write("## Input Info")
        st.write(f"Word Count: {word_count}")
        st.write(f"Sentence Count: {sentence_count}")
        st.write(f"Duration: {duration_sec:.2f} seconds")

        # Content & Structure
        sal_score = score_salutation(introduction)
        kw_score, mand_score, good_score = score_keywords(introduction)
        flow_score = check_flow(introduction)
        content_score = min(sal_score + kw_score + flow_score, 40)

        # Speech Rate
        speech_score, wpm = score_speech_rate(word_count, duration_sec)

        # Language & Grammar
        grammar_score, grammar_errors_per_100 = score_grammar(introduction, word_count)
        vocab_score, ttr = score_vocabulary(introduction, word_count)

        # Clarity
        clarity_score, filler_rate = score_clarity(introduction, word_count)

        # Engagement
        engagement_score, positive_sentiment = score_engagement(introduction)

        total_score = content_score + speech_score + grammar_score + vocab_score + clarity_score + engagement_score

        st.write("## Scores Breakdown")
        st.metric(label="Total Score", value=f"{total_score:.2f} / 100")

        st.subheader("Content & Structure")
        st.write(f"Salutation Level Score: {sal_score} / 5")
        st.write(f"Mandatory Keywords Score: {mand_score} / 20")
        st.write(f"Good-to-have Keywords Score: {good_score} / 10")
        st.write(f"Flow Score: {flow_score} / 5")

        st.subheader("Speech Rate")
        st.write(f"Speech Rate Score: {speech_score} / 10 (Words per minute: {wpm:.1f} WPM)")

        st.subheader("Language & Grammar")
        st.write(f"Grammar Score: {grammar_score} / 10 (Errors per 100 words: {grammar_errors_per_100:.2f})")
        st.write(f"Vocabulary Richness Score: {vocab_score} / 10 (TTR: {ttr:.2f})")

        st.subheader("Clarity")
        st.write(f"Clarity Score: {clarity_score} / 15 (Filler word rate: {filler_rate:.2f}%)")

        st.subheader("Engagement")
        st.write(f"Engagement Score: {engagement_score} / 15 (Positive sentiment score: {positive_sentiment:.2f})")

if __name__ == "__main__":
    main()