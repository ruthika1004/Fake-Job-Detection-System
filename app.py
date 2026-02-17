from flask import Flask, request, render_template_string
import re
import joblib
import random

app = Flask(__name__)

# ---------------- Load Model ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- Text Cleaning ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z₹0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------- Credibility Score ----------------
def calculate_credibility(text):
    score = 0
    if len(text.split()) > 80:
        score += 30
    if any(x in text for x in ["salary", "₹", "lpa"]):
        score += 20
    if any(x in text for x in ["experience", "skills", "responsibilities"]):
        score += 20
    if any(x in text for x in ["company", "technologies", "pvt", "ltd"]):
        score += 15
    if any(x in text for x in ["portal", "website", "career", "email"]):
        score += 15
    return score

# ---------------- Human Explanation ----------------
def generate_human_explanation(text, prediction):
    trust, risk = [], []

    if any(x in text for x in ["pvt", "ltd", "company", "technologies"]):
        trust.append("the company behind the role is clearly mentioned")
    if any(x in text for x in ["python", "sql", "java", "api", "rest"]):
        trust.append("specific technical skills are clearly defined")
    if any(x in text for x in ["salary", "₹", "lpa"]):
        trust.append("the salary details appear realistic")
    if any(x in text for x in ["portal", "website", "career"]):
        trust.append("a formal application process is described")

    if any(x in text for x in ["urgent", "immediate", "earn", "quick", "no experience"]):
        risk.append("the language suggests urgency or unrealistic promises")
    if not any(x in text for x in ["company", "pvt", "ltd"]):
        risk.append("no verifiable company information is provided")
    if len(text.split()) < 60:
        risk.append("the job description lacks sufficient detail")

    if prediction == 0:
        explanation = "This job appears genuine because " + ", ".join(trust) + "."
        if risk:
            explanation += " However, minor concerns were observed, such as " + ", ".join(risk) + "."
    else:
        explanation = "This job has been flagged as fake because " + ", ".join(risk) + "."
        if trust:
            explanation += " Nevertheless, some positive indicators were present, including " + ", ".join(trust) + "."

    return explanation

# ---------------- UI ----------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>AI-Powered Fake Job Detection</title>
<style>
body {
    font-family: Arial, sans-serif;
    background:#e9f0f7;
    color:#000;
}

.container {
    max-width: 900px;
    margin: 40px auto;
}

h1 {
    text-align:center;
    margin-bottom:30px;
}

.input-box {
    background:white;
    padding:20px;
    border-radius:12px;
    border:1px solid #cbd5e1;
}

textarea {
    width:100%;
    height:160px;
    padding:14px;
    font-size:16px;
    border-radius:8px;
    border:2px solid #000;
    box-sizing:border-box;
}

.buttons {
    margin-top:15px;
}

button {
    padding:10px 22px;
    font-size:15px;
    border:none;
    border-radius:8px;
    cursor:pointer;
}

.check {
    background:#2563eb;
    color:white;
}

.clear {
    background:#6b7280;
    color:white;
    margin-left:10px;
}

.result-section {
    margin-top:30px;
}

.result-label {
    font-size:22px;
    font-weight:bold;
    color:black;
}

.fake { color:#dc2626; }
.real { color:#16a34a; }

.score-title {
    margin-top:15px;
    font-size:18px;
    font-weight:bold;
}

.score-value {
    font-size:18px;
    font-weight:bold;
}

.explanation-title {
    margin-top:20px;
    font-size:18px;
    font-weight:bold;
}

.explanation-text {
    margin-top:8px;
    font-size:16px;
    line-height:1.7;
}
.score-row {
    margin-top:15px;
    display:flex;
    align-items:center;
    gap:10px;
}

.score-label {
    font-size:18px;
    font-weight:bold;
}

.score-value {
    font-size:18px;
    font-weight:bold;
    color:#2563eb; /* professional navy */
}

</style>
</head>

<body>
<div class="container">

<h1>AI-Powered Fake Job Detection</h1>

<form method="post">
<div class="input-box">
<textarea name="job_text" placeholder="Post your job description here...">{{ job_text }}</textarea>

<div class="buttons">
<button class="check" name="action" value="check">Check Job</button>
<button class="clear" name="action" value="clear">Clear</button>
</div>
</div>
</form>

{% if show_output %}
<div class="result-section">
<div class="result-label">
Result:
<span class="{{ color }}">{{ result }}</span>
</div>

<div class="score-row">
    <div class="score-label">Credibility Score:</div>
    <div class="score-value">{{ score }}/100</div>
</div>

<div class="explanation-title">Explanation:</div>
<div class="explanation-text">
{{ explanation }}
</div>
</div>
{% endif %}

</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = color = score = explanation = None
    job_text = ""
    show_output = False

    if request.method == "POST":
        action = request.form.get("action")
        job_text = request.form.get("job_text", "")

        if action == "check" and job_text.strip():
            clean = clean_text(job_text)
            prediction = model.predict(vectorizer.transform([clean]))[0]

            result = "FAKE JOB ❌" if prediction == 1 else "GENUINE JOB ✅"
            color = "fake" if prediction == 1 else "real"
            score = calculate_credibility(clean)
            explanation = generate_human_explanation(clean, prediction)
            show_output = True

        elif action == "clear":
            job_text = ""
            result = color = score = explanation = None
            show_output = False

    return render_template_string(
        HTML_PAGE,
        result=result,
        color=color,
        score=score,
        explanation=explanation,
        job_text=job_text,
        show_output=show_output
    )

if __name__ == "__main__":
    app.run(debug=True)
