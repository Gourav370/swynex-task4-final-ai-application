import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="SWYNEX AI Support Assistant", page_icon="🤖", layout="centered")

SAMPLE_DATA = [
    ("I cannot log in to my account", "Account Access", "High"),
    ("Please reset my password", "Account Access", "Medium"),
    ("My account has been locked", "Account Access", "High"),
    ("I was charged twice for my subscription", "Billing", "High"),
    ("Please explain the invoice on my bill", "Billing", "Medium"),
    ("I need a refund for my last payment", "Billing", "High"),
    ("The app crashes when I open it", "Technical Issue", "High"),
    ("The website is loading very slowly", "Technical Issue", "Medium"),
    ("I see an error message on the dashboard", "Technical Issue", "High"),
    ("How can I change my profile picture", "General Question", "Low"),
    ("Where can I find the user guide", "General Question", "Low"),
    ("What features are included in my plan", "General Question", "Low"),
    ("I want to cancel my subscription", "Account Management", "Medium"),
    ("Please update my email address", "Account Management", "Medium"),
    ("I cannot update my personal details", "Account Management", "Medium"),
    ("My payment failed but money was deducted", "Billing", "High"),
    ("The mobile app freezes during checkout", "Technical Issue", "High"),
    ("I forgot my username and password", "Account Access", "High"),
    ("Can you tell me your support hours", "General Question", "Low"),
    ("I want to change my subscription plan", "Account Management", "Medium"),
    ("I suspect someone accessed my account", "Account Access", "High"),
    ("Please send me a copy of my receipt", "Billing", "Low"),
    ("The page displays a server error", "Technical Issue", "High"),
    ("How do I add a new user", "General Question", "Low"),
]

@st.cache_resource
def build_model():
    df = pd.DataFrame(SAMPLE_DATA, columns=["query", "category", "priority"])
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(df["query"], df["category"])
    return model, df

model, training_data = build_model()

st.title("🤖 SWYNEX AI Support Assistant")
st.write("A beginner-friendly NLP prototype that classifies customer support queries and suggests a priority level.")
st.info("Prototype only: predictions are suggestions and should be checked by a human support agent.")

query = st.text_area(
    "Enter a customer query",
    placeholder="Example: I was charged twice and need help with my payment",
    height=120
)

if st.button("Classify query", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a customer query first.")
    else:
        category = model.predict([query.strip()])[0]
        text = query.lower()
        high_signals = ["charged twice", "money was deducted", "payment failed", "locked", "cannot log in",
                        "can't log in", "hacked", "unauthorized", "security", "urgent", "error", "crashes",
                        "crash", "not working", "suspicious", "refund"]
        medium_signals = ["cancel", "change", "update", "slow", "subscription", "reset", "cannot update"]
        if any(term in text for term in high_signals):
            priority = "High"
        elif any(term in text for term in medium_signals):
            priority = "Medium"
        else:
            priority = "Low"

        st.subheader("Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Suggested category", category)
        with col2:
            st.metric("Suggested priority", priority)
        st.caption("Category is predicted by a TF-IDF + Logistic Regression model. Priority is assigned using simple keyword rules.")
        st.write("**Recommended next step:** Review the prediction and route the query to the appropriate support queue.")

with st.expander("About this project"):
    st.markdown("""
    **Problem:** Customer support teams receive many queries and need to sort them quickly.

    **Method:** TF-IDF converts text into numerical features; Logistic Regression predicts a query category. A transparent keyword rule provides a priority suggestion.

    **Categories:** Account Access, Billing, Technical Issue, General Question, and Account Management.
    """)

with st.expander("Sample training data"):
    st.dataframe(training_data, use_container_width=True)

st.divider()
st.caption("SWYNEX AI Problem Design Project | Educational demonstration")
