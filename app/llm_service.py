import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY",
    "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"
)
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL      = "meta/llama-3.1-70b-instruct"

SYSTEM_PROMPT = """You are an expert AI assistant for a whole chicken without skin (poultry) business in India.
Your job is to answer questions about pricing, demand forecasting, stock estimation, and market trends.

Key pricing rules:
- Base price range: ₹210–₹270 per kg
- Demand > stock  → price increases by +5%
- Demand < 50% of stock → price decreases by -10%
- Festivals like Holi, Diwali, Dussehra, Independence Day, Christmas boost demand significantly
- Rainy weather slightly reduces footfall vs sunny days
- Peak sales hours: 1 PM (lunch) and 6 PM (evening)
- Seasons in this business: winter, spring, summer, monsoon, autumn

Always be concise (2–4 sentences). Give specific ₹ price estimates when asked about future prices.
Respond in a helpful, professional tone."""


def ask_llm(question: str, sales_summary: str) -> str:
    """Call NVIDIA Qwen3.5 API and return the AI response (streaming collected)."""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Current sales data context:\n{sales_summary}\n\n"
                    f"User question: {question}"
                )
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.60,
        "top_p": 0.95,
        "stream": True
    }

    try:
        # Provide a 10 second connection timeout and a 120 second read timeout
        response = requests.post(
            INVOKE_URL, headers=headers, json=payload,
            stream=True, timeout=(10, 120)
        )
        response.raise_for_status()

        full_text = ""
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    full_text += content
            except json.JSONDecodeError:
                continue

        # Return the collected text
        return full_text.strip() or "I couldn't generate a response. Please try again."

    except requests.exceptions.Timeout:
        return "The AI took too long to respond. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"API error ({e.response.status_code}): {e.response.text[:200]}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
