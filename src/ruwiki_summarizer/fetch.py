from typing import Optional
import requests

from .config import RUWIKI_HOST, RUWIKI_USER_AGENT


BASE_URL = f"https://{RUWIKI_HOST}/w/index.php"


def fetch_wikitext(title: str) -> Optional[str]:
    """
    Загружаем wikitext статьи с RuWiki через механизм action=raw
    (без MediaWiki API, через него не вышло достать).

    Пример:
        https://ru.ruwiki.ru/w/index.php?title=Изотопы&action=raw

    Получаем:
        - строку с исходным wikitext, если статья найдена;
        - None — при ошибке или отсутствии текста.
    """

    params = {
        "title": title,
        "action": "raw",
    }
    headers = {
        "User-Agent": RUWIKI_USER_AGENT,
    }

    try:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=15)
    except requests.RequestException as exc:
        print(f"[fetch] Ошибка сети при запросе статьи '{title}': {exc}")
        return None

    if response.status_code != 200:
        print(f"[fetch] Ошибка HTTP {response.status_code} для '{title}': {response.url}")
        return None

    wikitext = response.text.strip()

    if not wikitext:
        print(f"[fetch] Пустой wikitext для '{title}' (URL: {response.url})")
        return None

    return wikitext