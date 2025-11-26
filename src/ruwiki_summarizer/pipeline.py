from typing import Optional

from .fetch import fetch_wikitext
from .parse import extract_intro_and_sections, count_words
from .prompts import build_intro_prompts, build_section_prompts, decide_sentence_bounds
from .llm_client import LLMClient


def run_pipeline(
    title: str,
    backend: str = "local",
) -> Optional[str]:
    """
    Полный пайплайн генерации упрощённого конспекта статьи RuWiki.
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

    client = LLMClient(
        backend=backend,
        model_name="qwen2.5:3b-instruct",
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

        # Небольшой запас по токенам, чтобы текст не обрывался.
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
