import requests
from typing import Optional


class LLMClient:
    """
    Обёртка над локальной LLM через Ollama.
    Сейчас поддерживается только backend='local' и HTTP-эндпоинт
    http://localhost:11434/api/generate.
    """

    def __init__(
        self,
        backend: str = "local",
        model_name: str = "qwen2.5:3b-instruct",
    ):
        if backend != "local":
            raise ValueError("Сейчас поддерживается только backend='local' (Ollama).")

        self.backend = backend
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
    ) -> str:

        full_prompt = (
            f"<system>\n{system_prompt}\n</system>\n"
            f"<user>\n{user_prompt}\n</user>"
        )

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        if top_k is not None:
            payload["options"]["top_k"] = top_k

        resp = requests.post(self.api_url, json=payload, timeout=120)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        return data.get("response", "")