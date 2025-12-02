import json
from typing import Optional
from pathlib import Path

from .fetch import fetch_wikitext
from .parse import extract_intro_and_sections, count_words, basic_cleanup
from .prompts import build_intro_prompts, build_section_prompts, decide_sentence_bounds
from .fewshot import build_ege_fewshot_messages
from .llm_client import LLMClient
from .postprocess import clean_model_artifacts

def run_pipeline(
    title: str,
    backend: str = "local",
    mode: str = "per_section",
) -> Optional[str]:
    """
    Новый пайплайн:
    - mode='per_section': текущая логика по вступлению и разделам.
    - mode='ege_fewshot': один вызов chat-модели с few-shot примером ЕГЭ/ОГЭ.
    """
    if mode == "per_section":
        return _run_pipeline_per_section(title=title, backend=backend)
    elif mode == "ege_fewshot":
        return _run_pipeline_ege_fewshot(title=title, backend=backend)
    else:
        raise ValueError(f"Неизвестный режим mode={mode}")

def _run_pipeline_per_section(
    title: str,
    backend: str = "local",
) -> Optional[str]:
    """
    Старая логика (mode='per_section') генерации упрощённого конспекта статьи RuWiki.
    Шаги:
      1. Загрузка wikitext по названию статьи.
      2. Разбиение на вступление и именованные разделы.
      3. Формирование промптов и генерация упрощённого вступления.
      4. Формирование промптов и генерация конспекта для каждого раздела.
      5. Сборка итоговой статьи с сохранением исходной структуры заголовков.
    """
    wikitext = fetch_wikitext(title)
    if wikitext is None:
        print(f"Страница '{title}' не найдена.")
        return None

    intro_text, sections = extract_intro_and_sections(wikitext)

    print("Найденные разделы статьи:")
    for section_title in sections.keys():
        print(f"  - {section_title}")

    print(f"Длина вступления (слов): {count_words(intro_text)}")
    for section_title, section_text in sections.items():
        print(f"Длина раздела '{section_title}' (слов): {count_words(section_text)}")


    """
    Оставляю возможность использовать для локального backend-а Ollama + qwen2.5:3b-instruct,
    Для openrouter — модель qwen/qwen3-4b:free или mistralai/mistral-7b-instruct:free или google/gemma-3-12b-it:free или mistralai/mistral-small-3.1-24b-instruct:free.
    """

    model_name = (
        "qwen2.5:3b-instruct"
        if backend == "local"
        else "mistralai/mistral-small-3.1-24b-instruct:free"
    )

    client = LLMClient(
        backend=backend,
        model_name=model_name,
    )

    parts: list[str] = []

    # 1) Обработка вступления (без заголовка)
    if intro_text:
        min_s, max_s = 2, 3
        print(f"Вступление: целевой диапазон предложений: {min_s}–{max_s}")
        system_prompt, user_prompt = build_intro_prompts(intro_text, min_s, max_s)

        # Небольшой запас по токенам, чтобы текст не обрывался.
        max_tokens = max_s * 80

        intro_summary = client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
        ).strip()

        parts.append(intro_summary + "\n")

    # 2) Обработка каждого раздела по отдельности
    for section_title, section_text in sections.items():
        min_s, max_s = decide_sentence_bounds(section_text)
        print(
            f"Раздел '{section_title}': целевой диапазон предложений: "
            f"{min_s}–{max_s}"
        )

        system_prompt, user_prompt = build_section_prompts(
            section_title=section_title,
            section_text=section_text,
            min_sent=min_s,
            max_sent=max_s,
        )

        # Чуть увеличенный лимит токенов на случай списков и сложных формулировок.
        max_tokens = max_s * 100

        section_summary = client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
        ).strip()

        section_block = f"== {section_title} ==\n{section_summary}\n"
        parts.append(section_block)

    final_article = "\n\n".join(parts).strip()
    return final_article

def _run_pipeline_ege_fewshot(
    title: str,
    backend: str = "openrouter",
) -> Optional[str]:
    """
    Режим: одна генерация статьи целиком в стиле ЕГЭ/ОГЭ
    через chat few-shot (пример 'Изотопы' -> 'Изотопы (ЕГЭ-ОГЭ)').
    Работает только с backend='openrouter'.
    """
    if backend != "openrouter":
        raise ValueError("Режим ege_fewshot поддерживается только с backend='openrouter'.")

    wikitext = fetch_wikitext(title)
    if wikitext is None:
        print(f"Страница '{title}' не найдена.")
        return None

    # Очищаем статью до plain-text (без таблиц, шаблонов и т.п.)
    full_text = basic_cleanup(wikitext)
    print(f"Длина исходного текста (слов): {count_words(full_text)}")

    # Собираем chat-историю с примером БЫЛО/СТАЛО
    messages = build_ege_fewshot_messages(full_text, title)

    Path("logs").mkdir(exist_ok=True)

    with open("logs/last_messages.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    # Для OpenRouter используем мощную модель
    client = LLMClient(
        backend="openrouter",
        model_name="mistralai/mistral-small-3.1-24b-instruct:free",
    )

    # Один вызов chat-модели
    result = client.generate_chat(
        messages=messages,
        max_tokens=1800,
        temperature=0.3,
        top_p=0.9,
    )
    result = clean_model_artifacts(result)

    return result
