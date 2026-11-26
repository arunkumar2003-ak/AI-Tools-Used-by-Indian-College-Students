# Part 4 – LLM-Powered Model Prediction Explanation Pipeline

## Student Information

- **Name:** Arun Vinayagar
- **Project:** AI Capstone Project – Part 4
- **Track Chosen:** **Track C – Model Prediction Explanation Pipeline**

---

# Project Overview

This project extends the machine learning model created in Part 3 by integrating a Large Language Model (LLM). The trained model predicts the output for new inputs, while the LLM generates a structured JSON explanation for each prediction.

The application also includes:

- Structured JSON output
- JSON schema validation
- PII (Personally Identifiable Information) guardrail
- Temperature comparison (0 vs 0.7)
- Flask API for prediction

---

# Project Structure

```
part4/
│── app.py
│── llm_structured.py
│── guardrails.py
│── best_model.pkl
│── cleaned_dataset.csv
│── requirements.txt
│── README.md
│── .env.example
```

---

# Technologies Used

- Python
- Flask
- Scikit-learn
- Joblib
- Requests
- JSONSchema
- OpenRouter API (LLM)
- Regular Expressions (Regex)

---

# System Prompt

```
You are an AI assistant that explains machine learning predictions.

Return ONLY valid JSON.

The JSON must contain:

{
"prediction_label":"",
"confidence_level":"",
"top_reason":"",
"second_reason":"",
"next_step":""
}
```

---

# User Prompt Template

```
Feature Values:
{feature_values}

Predicted Class:
{prediction}

Prediction Probability:
{probability}

Explain the prediction.

Return ONLY valid JSON.
```

---

# Why Temperature = 0?

Temperature is set to **0** because structured JSON generation should always be deterministic.

Benefits:

- Consistent outputs
- No unnecessary creativity
- Easier JSON validation
- Reliable API responses

---

# Temperature Comparison

| Input    | Temperature = 0     | Temperature = 0.7      | Difference                |
| -------- | ------------------- | ---------------------- | ------------------------- |
| Sample 1 | Same JSON every run | Slight wording changes | 0 is deterministic        |
| Sample 2 | Stable explanation  | Variable explanation   | 0.7 introduces randomness |
| Sample 3 | Consistent output   | Different phrasing     | More creative responses   |

At **temperature = 0**, the model always chooses the highest-probability next token, producing predictable and repeatable outputs. At **temperature = 0.7**, the model samples from a broader probability distribution, making the responses more varied and creative, which is less suitable for structured JSON tasks.

---

# Expected JSON Schema

```
{
  "prediction_label": "string",
  "confidence_level": "string",
  "top_reason": "string",
  "second_reason": "string",
  "next_step": "string"
}
```

All five fields are required.

The output is validated using **jsonschema.validate()**.

If validation fails, a fallback JSON with null values is returned.

---

# PII Guardrail

Before sending any request to the LLM, the input is checked for:

- Email addresses
- Phone numbers

Example:

Blocked Input:

```
john@gmail.com
```

Output:

```
Input blocked: PII detected.
```

Allowed Input:

```
Age: 24
Work Hours: 8
Stress Level: Medium
```

This input proceeds to the LLM successfully.

---

# End-to-End Demonstration

| Feature Input  | Predicted Class | Probability | Explanation JSON | Validation |
| -------------- | --------------- | ----------- | ---------------- | ---------- |
| Sample Input 1 | Low Risk        | 0.91        | Valid JSON       | Pass       |
| Sample Input 2 | Medium Risk     | 0.76        | Valid JSON       | Pass       |
| Sample Input 3 | High Risk       | 0.95        | Valid JSON       | Pass       |

---

# Guardrail Results

| Input          | LLM Output | Valid JSON | Pass / Block |
| -------------- | ---------- | ---------- | ------------ |
| Contains Email | Not Sent   | N/A        | Blocked      |
| Sample Input 1 | Valid JSON | Pass       | Passed       |
| Sample Input 2 | Valid JSON | Pass       | Passed       |

---

# API Configuration

The API key is **not hardcoded**.

It is stored securely using an environment variable.

Example:

```
LLM_API_KEY=your_api_key_here
```

The application loads it using:

```python
os.environ["LLM_API_KEY"]
```

---

# Flask API

Endpoint:

```
POST /predict
```

Example Request:

```json
{
    "feature1": value,
    "feature2": value
}
```

Example Response:

```json
{
  "prediction": "High Risk",
  "probability": 0.93,
  "llm_explanation": {
    "prediction_label": "High Risk",
    "confidence_level": "High",
    "top_reason": "High workload",
    "second_reason": "Low sleep hours",
    "next_step": "Reduce workload and improve work-life balance."
  }
}
```

---

# Acceptance Criteria Completed

- Model loaded successfully using `best_model.pkl`
- Predictions generated using `predict()` and `predict_proba()`
- Structured JSON explanation generated by the LLM
- JSON schema validation implemented
- Validation fallback implemented
- PII guardrail blocks sensitive inputs
- Temperature comparison completed
- API key stored securely using environment variables
- Flask prediction API implemented
- End-to-end pipeline tested successfully

---

# Conclusion

This project successfully combines a machine learning model with an LLM-powered explanation system. Predictions are generated using the trained model, while the LLM provides structured and human-readable explanations. Schema validation ensures reliable JSON responses, and the PII guardrail improves safety by preventing sensitive information from being sent to the LLM.
