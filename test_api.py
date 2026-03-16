import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Accept": "text/event-stream"
}

payload = {
  "model": "qwen/qwen3.5-122b-a10b",
  "messages": [{"role":"user","content":"Hello"}],
  "max_tokens": 100,
  "temperature": 0.60,
  "top_p": 0.95,
  "stream": True,
  "chat_template_kwargs": {"enable_thinking":True},
}

print("Sending request...")
try:
    response = requests.post(INVOKE_URL, headers=headers, json=payload, stream=True, timeout=(10, 120))
    print(f"Status Code: {response.status_code}")
    
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
except Exception as e:
    print(f"Error: {e}")

