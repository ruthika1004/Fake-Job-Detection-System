# explainable_ai.py
# Rule-Based Explainable AI for Fake Job Detection
# This file explains WHY a job is classified as Fake or Genuine

import re

# ---------------- Text Cleaning ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z₹0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------- Explainable AI Logic ----------------
def generate_human_explanation(text, prediction):
    """
    Generates a human-readable explanation
    based on detected trust and risk patterns.
    """

    trust_signals = []
    risk_signals = []

    # Trust indicators
    if any(x in text for x in ["pvt", "ltd", "company", "technologies", "limited"]):
        trust_signals.append("the company behind the role is clearly mentioned")

    if any(x in text for x in ["python", "sql", "java", "api", "rest"]):
        trust_signals.append("specific technical skills are clearly defined")

    if any(x in text for x in ["salary", "₹", "lpa"]):
        trust_signals.append("the salary information appears realistic")

    if any(x in text for x in ["portal", "website", "career", "email"]):
        trust_signals.append("a formal application process is provided")

    # Risk indicators
    if any(x in text for x in ["urgent", "immediate", "earn", "quick", "no experience"]):
        risk_signals.append("the language suggests urgency or unrealistic promises")

    if not any(x in text for x in ["company", "pvt", "ltd", "technologies"]):
        risk_signals.append("no verifiable company information is provided")

    if len(text.split()) < 60:
        risk_signals.append("the job description lacks sufficient detail")

    # ---------------- Final Explanation ----------------
    if prediction == 0:  # Genuine Job
        explanation = "This job appears genuine because "
        if trust_signals:
            explanation += ", ".join(trust_signals) + "."
        else:
            explanation += "the overall structure follows standard hiring practices."

        if risk_signals:
            explanation += " However, minor concerns were observed, such as " + ", ".join(risk_signals) + "."

    else:  # Fake Job
        explanation = "This job has been flagged as fake because "
        if risk_signals:
            explanation += ", ".join(risk_signals) + "."
        else:
            explanation += "it shows patterns commonly associated with fraudulent postings."

        if trust_signals:
            explanation += " Nevertheless, some positive indicators were present, including " + ", ".join(trust_signals) + "."

    return explanation


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    sample_text = """
    We are urgently hiring data entry operators.
    Work from home opportunity.
    No experience required.
    Earn ₹25,000 per week.
    Immediate joining.
    """

    cleaned = clean_text(sample_text)

    # Example prediction (1 = Fake, 0 = Genuine)
    prediction = 1

    explanation = generate_human_explanation(cleaned, prediction)

    print("Prediction:", "FAKE JOB" if prediction == 1 else "GENUINE JOB")
    print("Explanation:", explanation)
