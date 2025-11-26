import re
from collections import OrderedDict
from typing import Tuple

import wikitextparser as wtp


AUXILIARY_SECTION_TITLES = {
    "см. также",
    "примечания",
    "литература",
    "источники",
    "ссылки",
    "внешние ссылки",
    "награды",
    "галерея",
    "дополнительно",
}


def normalize_title(title: str) -> str:
    """Приводим заголовок к нижнему регистру без лишних пробелов."""
    return (title or "").strip().lower()


def is_auxiliary_section(title: str) -> bool:
    """Проверяем, является ли раздел служебным (примечания, ссылки и т.п.)."""
    return normalize_title(title) in AUXILIARY_SECTION_TITLES


def strip_tables(text: str) -> str:
    """Удаляем таблицы wiki-формата: всё между {| и |}."""
    return re.sub(r"\{\|[\s\S]*?\|\}", "", text)


def strip_refs_and_templates(text: str) -> str:
    """
    Удаляет сноски <ref>...</ref>, self-closing <ref .../> и шаблоны {{...}}.
    Этого уровня очистки достаточно для подачи текста в LLMку.
    """
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    return text


def strip_links(text: str) -> str:
    """
    Упрощаем вики-ссылки:
    [[Статья|текст]] -> текст
    [[Статья]] -> Статья
    [https://example.com текст] -> текст
    """
    text = re.sub(r"\[\[(?:[^\|\]]*\|)([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\|\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    return text


def basic_cleanup(text: str) -> str:
    """
    Базовая очистка:
    - удаление таблиц, шаблонов и сносок;
    - упрощение ссылок;
    - нормализация пустых строк и пробелов.
    """
    text = strip_tables(text)
    text = strip_refs_and_templates(text)
    text = strip_links(text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def count_words(text: str) -> int:
    """Подсчитывает количество слов по пробельному разделению."""
    return len(text.split())


def extract_intro_and_sections(wikitext: str) -> Tuple[str, "OrderedDict[str, str]"]:
    """
    Разбивает статью RuWiki на вступление и именованные разделы.

    Возвращает:
      intro_text — очищенный текст вступления (до первого заголовка),
      sections   — OrderedDict{заголовок_раздела: очищенный_текст_раздела}.

    Служебные разделы (примечания, ссылки и т.п.) отбрасываются.
    """
    parsed = wtp.parse(wikitext)
    sections = parsed.sections

    intro_text = ""
    sections_dict: "OrderedDict[str, str]" = OrderedDict()

    if not sections:
        # fallback: нет явных разделов — чистим всю статью как единый текст
        cleaned = basic_cleanup(wikitext)
        return cleaned, sections_dict

    # Первый section без title — вступление
    first = sections[0]
    if first.title is None:
        intro_raw = first.contents
        intro_text = basic_cleanup(intro_raw)

    # Остальные — с заголовками
    for section in sections:
        title = section.title
        if not title:
            continue
        if is_auxiliary_section(title):
            continue

        clean_title = title.strip()
        if not clean_title:
            continue

        section_text = basic_cleanup(section.contents)
        if not section_text:
            continue

        sections_dict[clean_title] = section_text

    return intro_text, sections_dict