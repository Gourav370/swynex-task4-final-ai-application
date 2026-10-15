# SWYNEX AI Support Assistant — Task 4: Final AI Application

## Project overview
This beginner-friendly application demonstrates how Natural Language Processing (NLP) and machine learning can help a customer-support team sort incoming messages. It predicts a category for a customer query and suggests a priority level.

## Problem statement
Support teams receive repeated questions about accounts, billing, technical problems, and general product use. Manually sorting every message can take time. This prototype demonstrates a way to assist with initial triage.

## Features
- Enter a customer query in a simple web interface.
- Predict one of five categories:
  - Account Access
  - Billing
  - Technical Issue
  - General Question
  - Account Management
- Suggest High, Medium, or Low priority.
- View the sample training data inside the application.
- Provide a human-review reminder.

## Method
- **Text processing:** Scikit-learn `TfidfVectorizer` converts words and word pairs into numerical features.
- **Category model:** Logistic Regression learns patterns from a small example dataset.
- **Priority suggestion:** A transparent keyword-based rule provides an initial priority estimate.
- **Interface:** Streamlit provides the browser-based application.

The sample dataset is intentionally small and synthetic. It is suitable for demonstrating the workflow, not for production decisions or claims of high accuracy.

## Run locally
You need Python 3.10 or later.

1. Download or clone this repository.
2. Open Terminal in the project folder.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed in Terminal, usually `http://localhost:8501`.
6. Enter a customer query and select **Classify query**.

## Example test queries
- `I was charged twice for my subscription`
- `I cannot log in to my account`
- `The app crashes when I open it`
- `Where can I find the user guide?`

## Evaluation plan
For a more reliable version, collect a representative, consented, anonymized and human-labelled dataset. Split it into training and test sets, then measure:
- Category accuracy
- Precision, recall and F1-score for each category
- Confusion matrix
- Priority suggestions reviewed by support staff

No accuracy score is claimed in this demo because the included data is small and illustrative.

## Limitations
- The training examples are synthetic and few in number.
- Unusual, ambiguous, multilingual, or very short messages may be misclassified.
- Keyword priority rules can miss context, sarcasm, and urgency expressed in unfamiliar words.
- The model does not connect to a real helpdesk, account system, or live customer database.
- The application does not learn from user feedback automatically.

## Ethics, privacy and safety
- Do not enter passwords, payment-card details, private account data, or other sensitive personal information.
- Use anonymized data and obtain appropriate permission before training on real support conversations.
- Predictions are suggestions, not decisions. A human should review urgent or sensitive cases.
- Test performance across language varieties and customer groups to identify unfair errors.
- Provide a clear way for staff to correct predictions and escalate cases.
- Do not use this prototype as the sole basis for denying service or making consequential decisions.

## Repository structure
```text
SWYNEX_Task4_Final_AI_Application/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── sample_queries.csv
```

## Task 4 deliverable
This repository packages the problem statement, method, runnable demo, evaluation plan, limitations, and ethics notes. To complete the SWYNEX submission, upload this folder to GitHub, test the application, publish a LinkedIn explanation/demo post, and submit both URLs on the SWYNEX task page.

## Disclaimer
Educational prototype only. It is not a production-ready customer-support system.
