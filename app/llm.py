from openai import OpenAI
from app.config import settings
from app.prompts import fraud_explanation_prompt


class ExplanationService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model_name = "gpt-4o-mini"

        # Usage tracking (in-memory)
        self.total_llm_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def generate_explanation(
        self,
        fraud_probability: float,
        prediction: int,
        threshold: float,
    ):
        structured_summary = {
            "fraud_probability": fraud_probability,
            "prediction": prediction,
            "threshold": threshold,
        }

        prompt = fraud_explanation_prompt(structured_summary)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            # Increment call counter
            self.total_llm_calls += 1

            # Safely extract usage
            usage = response.usage

            if usage:
                self.total_prompt_tokens += usage.prompt_tokens or 0
                self.total_completion_tokens += usage.completion_tokens or 0
                self.total_tokens += usage.total_tokens or 0

            return response.choices[0].message.content.strip()

        except Exception:
            return "Explanation unavailable due to LLM service error."

    def get_llm_usage(self):
        return {
            "total_llm_calls": self.total_llm_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }