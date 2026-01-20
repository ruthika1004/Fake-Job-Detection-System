from flask import Flask, request, render_template_string
import re
import shap
import joblib

app = Flask(__name__)

# ---------------- Load Pretrained Model ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

explainer = None   # SHAP will be initialized only when needed


# ---------------- Text Cleaning ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# ---------------- Credibility Score ----------------
def calculate_credibility(text):
    score = 0
    if len(text) > 300: score += 30
    if "salary" in text or "₹" in text: score += 20
    if "experience" in text: score += 20
    if "company" in text: score += 15
    if "email" in text or "contact" in text: score += 15
    return score


# ---------------- UI ----------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Fake Job Detection</title>
<style>
body { font-family: Arial; background:#f4f6f8; }
.container { width:65%; margin:auto; background:white; padding:20px; border-radius:10px; }
textarea { width:100%; height:150px; }
button { padding:10px 20px; margin-right:10px; background:#007bff; color:white; border:none; border-radius:5px; }
.clear { background:#6c757d; }
.fake { color:red; font-weight:bold; }
.real { color:green; font-weight:bold; }
.score { font-size:18px; }
</style>
</head>
<body>

<div class="container">
<h2>AI-Powered Fake Job Detection System</h2>

<form method="post">
<textarea name="job_text" placeholder="Paste job description here...">{{ job_text }}</textarea><br><br>

<button type="submit" name="action" value="check">Check Job</button>
<button type="submit" name="action" value="clear" class="clear">Clear</button>
</form>

{% if result %}
<hr>
<h3>Result: <span class="{{ color }}">{{ result }}</span></h3>
<p class="score">Credibility Score: {{ score }}/100</p>

<h4>Explainable AI</h4>

<p><b style="color:red;">Words pushing towards FAKE:</b></p>
<ul>
{% for word in fake_words %}
<li style="color:red;">{{ word }}</li>
{% endfor %}
</ul>

<p><b style="color:green;">Words pushing towards GENUINE:</b></p>
<ul>
{% for word in genuine_words %}
<li style="color:green;">{{ word }}</li>
{% endfor %}
</ul>
{% endif %}

</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    global explainer

    result = color = score = None
    fake_words = []
    genuine_words = []
    job_text = ""

    if request.method == "POST":
        action = request.form.get("action")
        job_text = request.form.get("job_text", "")

        if action == "check" and job_text.strip() != "":
            clean = clean_text(job_text)
            vec = vectorizer.transform([clean])
            prediction = model.predict(vec)[0]

            result = "FAKE JOB ❌" if prediction == 1 else "GENUINE JOB ✅"
            color = "fake" if prediction == 1 else "real"
            score = calculate_credibility(clean)

            # Initialize SHAP only once (lazy loading)
            if explainer is None:
                explainer = shap.LinearExplainer(model, vec, feature_perturbation="interventional")

            shap_values = explainer.shap_values(vec)
            feature_names = vectorizer.get_feature_names_out()

            word_scores = sorted(
                zip(feature_names, shap_values[0]),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            for word, value in word_scores[:20]:
                if value > 0:
                    fake_words.append(word)
                else:
                    genuine_words.append(word)

            fake_words = fake_words[:5]
            genuine_words = genuine_words[:5]

        elif action == "clear":
            job_text = ""
            result = color = score = None
            fake_words = []
            genuine_words = []

    return render_template_string(
        HTML_PAGE,
        result=result,
        color=color,
        score=score,
        fake_words=fake_words,
        genuine_words=genuine_words,
        job_text=job_text
    )


if __name__ == "__main__":
    app.run(debug=True)
