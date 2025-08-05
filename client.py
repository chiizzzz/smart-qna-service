# مسیر فایل: client.py
import requests
import json

API_URL = "http://127.0.0.1:8000/api/v1/qna"


def ask_question(question_text):
    headers = {"Content-Type": "application/json"}
    payload = {"query": question_text}

    print(f"در حال ارسال سوال: '{question_text}'")

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


# سوال خود را اینجا بنویسید
my_question = "هزینه سفر در تپسی چگونه محاسبه می‌شود؟"
result = ask_question(my_question)

print("\n--- پاسخ دریافت شده از سرور ---")
print(json.dumps(result, indent=2, ensure_ascii=False))
