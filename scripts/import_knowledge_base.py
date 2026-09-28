"""
یک فایل JSON از پرسش و پاسخ‌ها را به پایگاه دانش ChromaDB وارد می‌کند.

فرمت فایل: لیستی از آبجکت‌ها که هر کدام حداقل کلیدهای "question" و "answer" را دارند
(مثلا vector_store.json قدیمی داخل "New folder/app.zip").

اجرا از ریشه پروژه:
    python -m scripts.import_knowledge_base path/to/file.json [--overwrite]
"""
import argparse
import json

from app.services.models.agent import get_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="مسیر فایل JSON")
    parser.add_argument("--overwrite", action="store_true", help="پایگاه دانش فعلی را کامل جایگزین کن")
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if isinstance(items, dict):
        items = items.get("data", [])

    qa_list = [
        {"question": item["question"], "answer": item["answer"]}
        for item in items
        if item.get("question") and item.get("answer")
    ]

    model = get_model()
    result = model.overwrite_database(qa_list) if args.overwrite else model.add_entries(qa_list)
    print(result["message"])


if __name__ == "__main__":
    main()
