import os
import ssl
import sys
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import httpx2

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
hf_model = os.getenv("HF_MODEL", "openai/gpt-oss-120b")

if not hf_token:
    raise RuntimeError("HF_TOKEN is missing. Add it to your .env file.")

client = InferenceClient(api_key=hf_token)

print("AI Flashcard Generator")
print("-------------------------")

topic = input("Enter your topic: ")

prompt = f"""
Create exactly 10 educational flashcards about: {topic}

For each flashcard, give:
Question: ...
Answer: ...

Keep the questions clear and useful for studying.
Number them from 1 to 10.
"""

try:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=hf_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            break
        except (httpx2.TransportError, ssl.SSLError):
            if attempt == 2:
                raise
            delay = 2 ** attempt
            print(f"Connection interrupted; retrying in {delay} seconds...")
            time.sleep(delay)
    result = response.choices[0].message.content

    print("\nFLASHCARDS")
    print("=" * 50)
    print(result)

except Exception as e:
    print("\n❌ Error:", e)