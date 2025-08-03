import os
import json
import requests

TOGETHER_API_KEY = "tgp_v1_bkDzfW5uXkhF5L5SgvcDPaAF5lN2DdzGEPYq1_y0Hk0"
LLM_MODEL = "meta-llama/Llama-3-70b-chat-hf"

SUPPORT_TAGS = [
    "پشتیبانی فنی",
    "فروش و قیمت‌گذاری",
    "مالی و صورتحساب",
    "حساب کاربری و ورود",
    "ارسال و تحویل",
    "پیشنهادات و انتقادات",
    "همکاری تجاری",
    "سوالات عمومی"
]

def generate_tags(user_query):
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
                "content": f"""شما یک سیستم طبقه‌بندی متن هستید. وظیفه شما انتخاب تگ‌های مرتبط با سوال کاربر از میان لیست زیر است.
فقط خروجی JSON آرایه‌ای بده، بدون توضیح اضافه.
تگ‌های مجاز: {json.dumps(SUPPORT_TAGS, ensure_ascii=False)}"""
            },
            {
                "role": "user",
                "content": f"سوال کاربر: \"{user_query}\""
            }
        ],
        "temperature": 0.0,
        "max_tokens": 100
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        tags = json.loads(content)
        if isinstance(tags, list):
            return tags
        return []
    except Exception as e:
        print(f"[خطا]: {e}")
        return []

def main():
    print("سیستم تگ‌گذاری سوالات (با LLaMA-3)")
    print("برای خروج 'exit' را وارد کنید.\n")

    while True:
        user_query = input("سوال: ").strip()
        if user_query.lower() == "exit":
            break

        tags = generate_tags(user_query)
        print(f"تگ‌های پیشنهادی: {tags}\n")

if __name__ == "__main__":
    main()
