# --- START OF FILE app.py (FIXED) ---

import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

# --- ۱. تنظیمات و بارگذاری مدل‌ها (یک بار در شروع سرور) ---

TOGETHER_API_KEY = "tgp_v1_bkDzfW5uXkhF5L5SgvcDPaAF5lN2DdzGEPYq1_y0Hk0"
LLM_MODEL = "meta-llama/Llama-3-70b-chat-hf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store.json"
SIMILARITY_THRESHOLD = 0.7

# (تغییر ۱) تعریف لیست تگ‌ها به عنوان یک ثابت سراسری
SUPPORT_TAGS = [
    "پشتیبانی فنی", "فروش و قیمت‌گذاری", "مالی و صورتحساب",
    "حساب کاربری و ورود", "ارسال و تحویل", "پیشنهادات و انتقادات",
    "همکاری تجاری", "سوالات عمومی"
]

# --- بارگذاری مدل‌ها (این بخش بدون تغییر باقی می‌ماند) ---
print("Flask App: در حال بارگذاری مدل امبدینگ...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Flask App: مدل امبدینگ با موفقیت بارگذاری شد.")

print("Flask App: در حال بارگذاری پایگاه دانش...")
with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)
all_doc_vectors = np.array([item["embedding"] for item in knowledge_base])
print(f"Flask App: پایگاه دانش با {len(knowledge_base)} آیتم بارگذاری شد.")

# --- ۲. توابع کمکی (کامل شده) ---

# (تغییر ۲) تابع حالا فقط یک ورودی می‌گیرد و از ثابت سراسری استفاده می‌کند
def generate_tags(user_query):
    """تگ‌ها را با استفاده از ثابت سراسری SUPPORT_TAGS تولید می‌کند."""
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": f"""شما یک سیستم طبقه‌بندی دقیق هستید. یک تگ مرتبط با سوال کاربر را از لیست زیر انتخاب کن. فقط خروجی JSON بده.
لیست تگ‌های مجاز: {json.dumps(SUPPORT_TAGS, ensure_ascii=False)}"""},
            {"role": "user", "content": f"سوال کاربر: \"{user_query}\""}
        ], "temperature": 0.0, "max_tokens": 100, "response_format": {"type": "json_object"}
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
    """پاسخ نهایی را تولید می‌کند."""
    if not context_docs:
        return "متاسفانه پاسخ مشخصی برای سوال شما در پایگاه دانش ما وجود ندارد."
    context_text = "\n\n---\n\n".join([f"سوال: {item['question']}\nپاسخ: {item['answer']}" for item in context_docs])
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": """شما یک کارشناس پشتیبانی حرفه‌ای هستید. با توجه به اطلاعات زیر به سوال کاربر به طور دقیق و خلاصه پاسخ بده. مستقیماً به سوال جواب بده."""},
            {"role": "user", "content": f"## اطلاعات مرتبط:\n{context_text}\n\n## سوال کاربر:\n\"{user_query}\""}
        ], "temperature": 0.2, "max_tokens": 500
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[خطا در تولید پاسخ]: {e}")
        return "متاسفانه در ارتباط با مدل هوش مصنوعی مشکلی پیش آمده است."


# --- ۳. هسته منطقی برنامه (بدون تغییر) ---
def process_query(user_query):
    # این تابع حالا به درستی کار خواهد کرد چون generate_tags دیگر به ورودی دوم نیاز ندارد
    tags = generate_tags(user_query)
    user_vector = embedder.encode(user_query)
    similarities = cosine_similarity([user_vector], all_doc_vectors)[0]
    matches = sorted([
        (knowledge_base[i], similarity) for i, similarity in enumerate(similarities) if similarity >= SIMILARITY_THRESHOLD
    ], key=lambda x: x[1], reverse=True)
    top_matches = [m[0] for m in matches[:3]]
    answer = generate_response(user_query, top_matches)
    return {"tags_identified": tags, "final_answer": answer, "retrieved_context_count": len(top_matches)}

# --- ۴. ساخت برنامه Flask و تعریف روت API (بدون تغییر) ---
app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>🧠 سرور QA فعال است.</h1><p>یک درخواست POST به /ask ارسال کنید.</p>"

@app.route('/ask', methods=['POST'])
def handle_ask_request():
    if not request.is_json or 'query' not in request.get_json():
        return jsonify({"error": "درخواست نامعتبر. باید JSON باشد و کلید 'query' را داشته باشد."}), 400
    user_query = request.get_json()['query']
    print(f"Flask App: دریافت درخواست برای سوال: '{user_query}'")
    try:
        result = process_query(user_query)
        print("Flask App: پاسخ تولید شد.")
        return jsonify(result)
    except Exception as e:
        print(f"Flask App: یک خطای داخلی رخ داد: {e}")
        return jsonify({"error": "یک خطای داخلی در سرور رخ داد."}), 500

# --- ۵. اجرای سرور ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# --- END OF FILE app.py (FIXED) ---