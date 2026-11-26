import os
import json
import re
import joblib
import requests
import numpy as np
from jsonschema import validate, ValidationError

API_KEY = os.getenv("LLM_API_KEY")
MODEL = "openai/gpt-4o-mini"
URL = "https://openrouter.ai/api/v1/chat/completions"

# Load Model
model = joblib.load("best_model.pkl")


def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


schema = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string"},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": [
        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step"
    ]
}


def call_llm(system_prompt, user_prompt, temperature=0):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": 512
    }

    response = requests.post(URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(response.status_code)
        return None

    return response.json()["choices"][0]["message"]["content"]


def explain_prediction(features):

    text = json.dumps(features)

    if has_pii(text):
        return {"error": "Input blocked: PII detected"}

    X = np.array([list(features.values())])

    pred = model.predict(X)[0]

    prob = float(model.predict_proba(X).max())

    system_prompt = """
You are an AI assistant.

Return ONLY valid JSON.

{
"prediction_label":"",
"confidence_level":"",
"top_reason":"",
"second_reason":"",
"next_step":""
}
"""

    user_prompt = f"""
Feature Values:
{text}

Prediction:
{pred}

Probability:
{prob}
"""

    output = call_llm(system_prompt, user_prompt, 0)

    if output is None:
        return None

    try:

        result = json.loads(output.strip())

        validate(result, schema)

        return result

    except (json.JSONDecodeError, ValidationError):

        return {
            "prediction_label": None,
            "confidence_level": None,
            "top_reason": None,
            "second_reason": None,
            "next_step": None
        }