# --- START OF FILE TagVDB.py ---

import json
import pandas as pd
import requests
from tqdm import tqdm

TOGETHER_API_KEY = "tgp_v1_bkDzfW5uXkhF5L5SgvcDPaAF5lN2DdzGEPYq1_y0Hk0"
LLM_MODEL = "meta-llama/Llama-3-70b-chat-hf"

SUPPORT_TAGS = [
    "پشتیبانی فنی", "فروش و قیمت‌گذاری", "مالی و صورتحساب",
    "حساب کاربری و ورود", "ارسال و تحویل", "پیشنهادات و انتقادات",
    "همکاری تجاری", "سوالات عمومی"
]

QUESTION_COLUMN = "سوال"
ANSWER_COLUMN = "پاسخ"


def generate_tags(text_to_classify):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": f"""شما یک سیستم طبقه‌بندی متن بسیار دقیق هستید. وظیفه شما انتخاب یک تگ مرتبط با متن ورودی از میان لیست زیر است. پاسخ شما باید فقط یک آرایه JSON معتبر باشد بدون هیچ متن اضافه‌ای. مثال: ["فروش و قیمت‌گذاری"]
لیست تگ‌های مجاز: {json.dumps(SUPPORT_TAGS, ensure_ascii=False)}"""
            },
            {"role": "user", "content": f"متن زیر را طبقه‌بندی کن:\n\"{text_to_classify}\""}
        ],
        "temperature": 0.0,
        "max_tokens": 100,
        "response_format": {"type": "json_object"}
    }

    content = ""  # تعریف اولیه برای جلوگیری از NameError
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        tags = json.loads(content)
        if isinstance(tags, list):
            return ", ".join(tags) if tags else "سوالات عمومی"
        else:
            return "فرمت نامعتبر"
    except json.JSONDecodeError:
        print(f"  [خطای JSON] پاسخ نامعتبر از مدل: {content}")
        return "خطای پارس JSON"
    except Exception as e:
        print(f"  [خطا در API] {e}")
        return "خطای API"


def main():
    input_file = "input.xlsx"
    output_file = "output_tagged_llama.xlsx"

    try:
        df = pd.read_excel(input_file)
        print(f"فایل '{input_file}' با موفقیت خوانده شد. تعداد ردیف‌ها: {len(df)}")
    except FileNotFoundError:
        print(f"خطا: فایل '{input_file}' یافت نشد.")
        return

    if not all(col in df.columns for col in [QUESTION_COLUMN, ANSWER_COLUMN]):
        print(f"خطا: ستون‌های لازم '{QUESTION_COLUMN}' و '{ANSWER_COLUMN}' یافت نشدند.")
        return

    tags_column = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="در حال تگ‌گذاری با Llama-3"):
        full_text = f"سوال: {str(row[QUESTION_COLUMN])}\nپاسخ: {str(row[ANSWER_COLUMN])}"
        tags = generate_tags(full_text)
        tags_column.append(tags)

    df["تگ‌های Llama3"] = tags_column
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\nپردازش تمام شد! فایل نهایی در '{output_file}' ذخیره شد.")


if __name__ == "__main__":
    main()

# --- END OF FILE TagVDB.py ---