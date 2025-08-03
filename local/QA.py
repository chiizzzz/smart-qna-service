import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# --- تنظیمات ---
TOGETHER_API_KEY = "tgp_v1_bkDzfW5uXkhF5L5SgvcDPaAF5lN2DdzGEPYq1_y0Hk0"
LLM_MODEL = "meta-llama/Llama-3-70b-chat-hf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store.json"  # فایل پایگاه دانش که از قبل باید ساخته شده باشد
SIMILARITY_THRESHOLD = 0.7  # آستانه شباهت کمی کاهش یافت تا موارد بیشتری شانس بررسی داشته باشند

# --- بارگذاری مدل امبدینگ (یک بار در شروع برنامه) ---
print("در حال بارگذاری مدل امبدینگ...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("مدل امبدینگ با موفقیت بارگذاری شد.")

# --- بارگذاری پایگاه دانش ---
try:
    with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
    print(f"پایگاه دانش از '{VECTOR_STORE_PATH}' با موفقیت بارگذاری شد. ({len(knowledge_base)} آیتم)")
except FileNotFoundError:
    print(f"خطای حیاتی: فایل پایگاه دانش '{VECTOR_STORE_PATH}' یافت نشد.")
    exit()
except json.JSONDecodeError:
    print(f"خطای حیاتی: فایل پایگاه دانش '{VECTOR_STORE_PATH}' فرمت JSON معتبری ندارد.")
    exit()


# --- توابع API با ساختار پیام بهبودیافته ---

def generate_tags(user_query, support_tags):
    """تگ‌ها را با استفاده از ساختار پیام بهینه (system/user) تولید می‌کند."""
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": f"""شما یک سیستم طبقه‌بندی متن هستید. وظیفه شما انتخاب تگ‌ مرتبط با سوال کاربر از میان لیست زیر است. فقط خروجی JSON آرایه‌ای بده، بدون توضیح اضافه.
تگ‌های مجاز: {json.dumps(support_tags, ensure_ascii=False)}"""
            },
            {
                "role": "user",
                "content": f"سوال کاربر: \"{user_query}\""
            }
        ],
        "temperature": 0.0,
        "max_tokens": 100,
        "response_format": {"type": "json_object"}  # مدل را مجبور به تولید JSON می‌کند
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        tags = json.loads(response.json()["choices"][0]["message"]["content"])
        return tags if isinstance(tags, list) else []
    except Exception as e:
        print(f"[خطای تگ‌گذاری]: {e}")
        return []


def generate_response(user_query, context_docs):
    """پاسخ را با استفاده از ساختار پیام بهینه (system/user) تولید می‌کند."""
    if not context_docs:
        return "متاسفانه پاسخ مشخصی برای سوال شما در پایگاه دانش ما وجود ندارد. لطفاً سوال خود را واضح‌تر بپرسید."

    context_text = "\n\n---\n\n".join(
        [f"مورد یافت شده {i + 1}:\nسوال: {item['question']}\nپاسخ: {item['answer']}" for i, item in
         enumerate(context_docs)]
    )

    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """شما یک کارشناس پشتیبانی بسیار حرفه‌ای و خوش‌برخورد هستید. وظیفه شما پاسخ دادن به سوال کاربر فقط و فقط بر اساس اطلاعاتی است که در اختیارتان قرار می‌گیرد.
- پاسخ باید محترمانه، دقیق و خلاصه باشد.
- مستقیماً به سوال جواب بده و از جملات اضافی مانند "با توجه به اطلاعاتی که دادید..." خودداری کن.
- اگر پاسخ در اطلاعات موجود نیست، به صراحت بگو: "متاسفانه اطلاعات دقیقی در این مورد در پایگاه دانش من وجود ندارد." """
            },
            {
                "role": "user",
                "content": f"## اطلاعات مرتبط از پایگاه دانش:\n{context_text}\n\n## سوال کاربر:\n\"{user_query}\""
            }
        ],
        "temperature": 0.2,  # کمی کاهش برای پاسخ قطعی‌تر
        "max_tokens": 500  # کمی افزایش برای پاسخ‌های کامل‌تر
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.strip()
    except Exception as e:
        print(f"[خطا در تولید پاسخ]: {e}")
        return "متاسفانه در ارتباط با مدل هوش مصنوعی مشکلی پیش آمده است."


def main():
    print("\n" + "=" * 30)
    print("🧠 سیستم پشتیبانی هوشمند آماده است.")
    print("   (برای خروج 'exit' را تایپ کنید)")
    print("=" * 30 + "\n")

    support_tags = ["پشتیبانی فنی", "فروش و قیمت‌گذاری", "مالی و صورتحساب", "حساب کاربری و ورود", "ارسال و تحویل",
                    "پیشنهادات و انتقادات", "همکاری تجاری", "سوالات عمومی"]

    # استخراج تمام بردارهای داکیومنت‌ها به صورت یکجا برای بهینه‌سازی
    all_doc_vectors = np.array([item["embedding"] for item in knowledge_base])

    while True:
        user_query = input("❓ سوال شما: ").strip()
        if user_query.lower() == "exit":
            print("موفق باشید!")
            break
        if not user_query:
            continue

        # ۱. تگ‌گذاری سوال (برای نمایش به کاربر یا فیلتر کردن در آینده)
        tags = generate_tags(user_query, support_tags)
        print(f"🏷️  تگ‌های شناسایی‌شده: {tags if tags else 'هیچ تگ مشخصی یافت نشد'}")

        # ۲. تولید امبدینگ سوال
        user_vector = embedder.encode(user_query)

        # ۳. محاسبه شباهت (بهینه‌سازی شده)
        similarities = cosine_similarity([user_vector], all_doc_vectors)[0]

        # ۴. پیدا کردن موارد مشابه و مرتب‌سازی
        matches = []
        for i, similarity in enumerate(similarities):
            if similarity >= SIMILARITY_THRESHOLD:
                matches.append((knowledge_base[i], similarity))

        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:3]]  # انتخاب ۳ مورد برتر

        # ۵. تولید پاسخ
        print("💬 در حال تولید پاسخ...")
        response = generate_response(user_query, top_matches)
        print(f"\n✅ پاسخ:\n{response}\n" + "-" * 30 + "\n")


if __name__ == "__main__":
    main()