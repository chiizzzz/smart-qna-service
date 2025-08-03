
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings


class QAModel:

    def __init__(self):
        self._knowledge_base = None
        self._all_doc_vectors = None
        self._embedder = None
        self._load_dependencies()

    def _load_dependencies(self):

        print("Service: در حال بارگذاری مدل امبدینگ...")
        self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

        print("Service: در حال بارگذاری پایگاه دانش برداری...")
        with open(settings.VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
            self._knowledge_base = json.load(f)
        self._all_doc_vectors = np.array([item["embedding"] for item in self._knowledge_base])
        print("Service: تمام مدل‌ها و داده‌ها با موفقیت بارگذاری شدند.")

    def _generate_tags(self, user_query: str) -> list:

        url = "https://api.together.xyz/v1/chat/completions"
        headers = {"Authorization": f"Bearer {settings.TOGETHER_API_KEY}"}
        payload = {
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": f"""شما یک سیستم طبقه‌بندی دقیق هستید. یک یا چند تگ مرتبط را از لیست زیر انتخاب کن. فقط خروجی JSON بده.
                لیست تگ‌های مجاز: {json.dumps(settings.SUPPORT_TAGS, ensure_ascii=False)}"""},
                {"role": "user", "content": f"سوال: \"{user_query}\""}
            ], "temperature": 0.0, "max_tokens": 100, "response_format": {"type": "json_object"}
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            tags = json.loads(response.json()["choices"][0]["message"]["content"])
            return tags if isinstance(tags, list) else []
        except Exception as e:
            print(f"[خطای تگ‌گذاری]: {e}")
            return []

    def _generate_response(self, user_query: str, context_docs: list) -> str:

        if not context_docs:
            return "نمیدانم."
        context_text = "\n\n---\n\n".join(
            [f"سوال: {item['question']}\nپاسخ: {item['answer']}" for item in context_docs])
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {"Authorization": f"Bearer {settings.TOGETHER_API_KEY}"}
        payload = {
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system",
                 "content": "شما یک کارشناس پشتیبانی حرفه‌ای هستید. با توجه به اطلاعات زیر به سوال کاربر به طور دقیق و خلاصه پاسخ بده."},
                {"role": "user", "content": f"## اطلاعات مرتبط:\n{context_text}\n\n## سوال کاربر:\n\"{user_query}\""}
            ], "temperature": 0.2, "max_tokens": 500
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=40)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[خطا در تولید پاسخ]: {e}")
            return "متاسفانه در ارتباط با مدل هوش مصنوعی مشکلی پیش آمده است."

    def predict(self, user_query: str) -> dict:
        tags = self._generate_tags(user_query)
        user_vector = self._embedder.encode(user_query)
        similarities = cosine_similarity([user_vector], self._all_doc_vectors)[0]
        matches = sorted([(self._knowledge_base[i], sim) for i, sim in enumerate(similarities) if
                          sim >= settings.SIMILARITY_THRESHOLD], key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:3]]
        answer = self._generate_response(user_query, top_matches)

        return {
            "tags_identified": tags,
            "final_answer": answer,
            "retrieved_context_count": len(top_matches)
        }
model_instance = QAModel()
def get_model():
    return model_instance