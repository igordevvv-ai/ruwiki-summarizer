import re

def clean_model_artifacts(text: str) -> str:
    """
    Удаляем системные артефакты, которые иногда прокидывает модель:
    - <s>, </s>
    - [OUT], [/OUT]
    - лишние пробелы вокруг них
    """
    if not text:
        return text

    # Убираем маркеры, которые иногда вставляют модели
    artifacts = ["<s>", "</s>", "[OUT]", "[/OUT]"]
    for a in artifacts:
        text = text.replace(a, "")

    # Убираем лишние пробелы, табы
    text = re.sub(r"[ \t]+", " ", text)

    # Чистим пробелы по краям каждой строки
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Общий strip в начале и в конце текста
    return text.strip()