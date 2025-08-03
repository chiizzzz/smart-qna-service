# --- START OF FILE client.py ---
import requests
import json

API_URL = "http://127.0.0.1:5000/ask"


def ask_question(question_text):
    """یک سوال به API ارسال کرده و پاسخ را برمی‌گرداند."""

    headers = {"Content-Type": "application/json"}
    payload = {"query": question_text}

    print(f"در حال ارسال سوال: '{question_text}'")

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=90)  # افزایش زمان انتظار
        response.raise_for_status()  # بررسی خطاهای HTTP
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "سرور در زمان تعیین شده پاسخ نداد. ممکن است در حال پردازش باشد."}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


# === سوال خود را اینجا وارد کنید ===
my_question = "چطور میتونم به راننده امتیاز بدم؟"

result = ask_question(my_question)

# چاپ نتیجه
print("\n--- پاسخ دریافت شده از سرور ---")
print(json.dumps(result, indent=2, ensure_ascii=False))
# --- END OF FILE client.py ---