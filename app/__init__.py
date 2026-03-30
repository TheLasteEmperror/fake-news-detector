from flask import Flask, render_template, request
import joblib

# Загружаем модель и векторизатор
model = joblib.load("../saved_model/logistic_model.pkl")
vectorizer = joblib.load("../saved_model/tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    if request.method == "POST":
        text = request.form["news_text"]
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        result = "Fake" if pred == 1 else "Real"
        confidence = round(float(max(proba)) * 100, 1)
    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)