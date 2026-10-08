"""One function the whole project uses to talk to an LLM."""
import requests
from . import config


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    if config.LLM_PROVIDER == "mock":
        from . import mock_llm
        return mock_llm.chat(system, user)
    resp = requests.post(
        f"{config.LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
        json={
            "model": config.LLM_MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
