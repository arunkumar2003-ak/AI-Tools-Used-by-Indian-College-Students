from flask import Flask, request, jsonify
from llm_structured import explain_prediction

app = Flask(__name__)

@app.route("/")
def home():
    return "LLM Structured API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    result = explain_prediction(data)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)