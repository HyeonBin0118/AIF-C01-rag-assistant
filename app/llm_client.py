import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")
        _client = OpenAI(api_key=api_key)
    return _client


def chat_completion(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.3) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()