import requests
from typing import Optional, List, Dict


class LLMClient:
    """
    Обёртка над LLM:
    - backend='local'  -> локальная модель через Ollama (HTTP: /api/generate)
    - backend='openrouter' -> внешняя модель через OpenRouter API (chat-completions)
    """

    def __init__(
        self,
        backend: str = "local",
        model_name: str = "qwen2.5:3b-instruct",
    ):
        self.backend = backend
        self.model_name = model_name

        if backend == "local":
            self.api_url = "http://localhost:11434/api/generate"

        elif backend == "openrouter":
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            # Ключ сейчас захардкодил.
            #self.api_key = "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"

        else:
            raise ValueError(f"Неподдерживаемый backend: {backend}")

    # ================== ЕДИНЫЙ СТАРЫЙ ИНТЕРФЕЙС ==================

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Единая точка входа.
        Внутри роутим либо на локальную Ollama, либо на OpenRouter.
        """
        if self.backend == "local":
            return self._generate_local(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        elif self.backend == "openrouter":
            return self._generate_openrouter(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            raise ValueError(f"Неподдерживаемый backend: {self.backend}")

    # ================== ЛОКАЛЬНАЯ МОДЕЛЬ (OLLAMA) ==================

    def _generate_local(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Вызов локальной модели через Ollama.
        """
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
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

        data = resp.json()
        return (data.get("response") or "").strip()

    # ================== OPENROUTER: старый интерфейс ==================

    def _generate_openrouter(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """
        Вызов модели через OpenRouter API (chat-completions), но в старом формате
        system + user → text.
        Внутри это просто один чат-запрос.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    # ================== OPENROUTER: chat (few-shot) ==================

    def generate_chat(
            self,
            messages: List[Dict[str, str]],
            max_tokens: Optional[int] = 512,
            temperature: float = 0.3,
            top_p: float = 0.9,
            max_retries: int = 3,
    ) -> str:
        """
        Chat-режим (few-shot): принимаем список сообщений.
        Работает только для backend='openrouter'.
        Добавлен автоматический повтор при:
          - HTTP 429 (rate limit)
          - пустом или некорректном ответе.
        """
        if self.backend != "openrouter":
            raise ValueError("generate_chat поддерживается только при backend='openrouter'.")

        import time  # важно!

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/igordevvv-ai/ruwiki-summarizer",
            "X-Title": "RuWiki Summarizer (few-shot EGE)",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        # ---- RETRY LOOP ----
        for attempt in range(max_retries):
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=180)

            # 1) RATE LIMIT (429)
            if resp.status_code == 429:
                wait = 2 * (attempt + 1)
                print(f"[Retry {attempt + 1}/{max_retries}] OpenRouter 429 — ждём {wait} сек...")
                time.sleep(wait)
                continue

            # 2) Другие ошибки
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

            data = resp.json()

            if "error" in data:
                wait = 2 * (attempt + 1)
                print(f"[Retry {attempt + 1}/{max_retries}] Ошибка в payload. Ждём {wait} сек...")
                time.sleep(wait)
                continue

            # 3) Достаём контент
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                wait = 2 * (attempt + 1)
                print(f"[Retry {attempt + 1}/{max_retries}] Некорректный формат ответа. Ждём {wait} сек...")
                time.sleep(wait)
                continue

            cleaned = (content or "").strip()

            # 4) Пустой ответ → retry
            if not cleaned:
                wait = 2 * (attempt + 1)
                print(f"[Retry {attempt + 1}/{max_retries}] Пустой ответ. Ждём {wait} сек...")
                time.sleep(wait)
                continue

            # ----- SUCCESS -----
            return cleaned

        # ---- после всех попыток ----
        raise RuntimeError(f"Модель не ответила после {max_retries} попыток.")
