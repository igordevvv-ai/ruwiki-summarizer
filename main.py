import argparse
import os
import sys

# Добавляем папку src в sys.path, чтобы можно было импортировать пакет ruwiki_summarizer
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ruwiki_summarizer.pipeline import run_pipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация упрощённого конспекта статьи RuWiki для школьников."
    )
    parser.add_argument(
        "--title",
        type=str,
        required=True,
        help="Название статьи в Рувики (как в заголовке страницы).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="local",
        choices=["local", "openrouter"],
        help="Тип LLM-бэкенда: локальная модель или OpenRouter API.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь к файлу, куда сохранить результат (если не задан, выводим в консоль).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="per_section",
        choices=["per_section", "ege_fewshot"],
        help="Режим работы пайплайна: по разделам ('per_section') или целиком в формате ЕГЭ/ОГЭ ('ege_fewshot').",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        title=args.title,
        backend=args.backend,
        mode=args.mode,
    )

    if summary is None:
        return

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Результат сохранён в {args.output}")
    else:
        print("\n==== СГЕНЕРИРОВАННЫЙ КОНСПЕКТ ====\n")
        print(summary)


if __name__ == "__main__":
    main()