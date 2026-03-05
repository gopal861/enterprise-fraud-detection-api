def fraud_explanation_prompt(structured_summary: dict) -> str:
    return f"""
You are explaining fraud detection results.

Rules:
- Do NOT change the decision.
- Do NOT invent new facts.
- Only explain the structured result.
- Keep explanation under 4 sentences.
- Be concise and factual.

Structured Result:
{structured_summary}
"""