import requests
import json

# --- مقادیر زیر را با اطلاعات خودتان جایگزین کنید ---
# این مقادیر را از فایل .env خود کپی کنید
API_KEY = "sk-2m6GN7v2vnvszYq5JlrVxLeZ44UbZDH7AojkQ65aLyRtWgOW"
BASE_URL = "https://new-api.atenatech.ir/"
MODEL_NAME = "gpt-4o"  # یا نام مدلی که می‌خواهید تست کنید
# ----------------------------------------------------

# آدرس کامل اندپوینت برای چت
# معمولاً آدرس‌ها به /v1/chat/completions ختم می‌شوند
url = f"{BASE_URL.strip('/')}/v1/chat/completions"

# هدرهای لازم برای ارسال درخواست، شامل کلید API
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# یک بدنه درخواست (payload) ساده برای تست
payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "سلام، آیا کار می‌کنی؟"
        }
    ],
    "max_tokens": 10
}

print(f"ارسال درخواست به: {url}")
print("..." * 20)

try:
    # ارسال درخواست به سرور
    response = requests.post(url, headers=headers, json=payload, timeout=20) # 20 ثانیه مهلت پاسخ

    # چاپ نتایج
    print(f"کد وضعیت (Status Code): {response.status_code}")
    print("--- هدرهای پاسخ (Response Headers) ---")
    print(response.headers)
    print("\n--- متن خام پاسخ (Raw Response Text) ---")
    print(response.text)
    print("---" * 15)

    # تلاش برای پارس کردن پاسخ به عنوان JSON
    try:
        response_json = response.json()
        print("\nپاسخ با موفقیت به عنوان JSON پارس شد.")
        print(response_json)
    except json.JSONDecodeError:
        print("\nخطا: پاسخ دریافت شده با فرمت JSON معتبر نیست.")

except requests.exceptions.RequestException as e:
    print(f"\nیک خطای ارتباطی رخ داد: {e}")