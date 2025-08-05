# مسیر فایل: app/services/models.py

import json
import numpy as np
import requests
import traceback
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings
from app.services.tools.tools import operator_tool  # <-- وارد کردن ابزار جدید
import threading  # <-- برای جلوگیری از تداخل در زمان نوشتن فایل
from openai import OpenAI

class QAModel:
    """
    این کلاس، کل RAG سیستم به همراه منطق مدیریت پایگاه دانش و حلقه بازخورد را پیاده‌سازی می‌کند.
    """

    def __init__(self):
        self._knowledge_base = None
        self._all_doc_vectors = None
        self._embedder = None
        self._openai_client = None
        self._file_lock = threading.Lock()  # <-- قفل برای جلوگیری از تداخل در دسترسی به فایل
        self._load_dependencies()

    def _load_dependencies(self):
        print("Service: در حال بارگذاری مدل امبدینگ...")
        self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        print("Service: در حال مقداردهی اولیه کلاینت OpenAI...")
        self._openai_client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )
        self._reload_knowledge_base()

    def _reload_knowledge_base(self):
        """پایگاه دانش را از فایل می‌خواند. این تابع باید هربار پس از تغییر فایل فراخوانی شود."""
        with self._file_lock:
            print("Service: در حال بارگذاری/بارگذاری مجدد پایگاه دانش برداری...")
            try:
                with open(settings.VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
                    self._knowledge_base = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self._knowledge_base = []  # اگر فایل وجود نداشت یا خالی بود

        if self._knowledge_base:
            self._all_doc_vectors = np.array([item["embedding"] for item in self._knowledge_base])
        else:
            self._all_doc_vectors = np.array([])
        print(f"Service: بارگذاری کامل شد. تعداد آیتم‌ها: {len(self._knowledge_base)}")

    def _save_knowledge_base(self):
        """تغییرات را در فایل vector_store.json ذخیره می‌کند."""
        with self._file_lock:
            with open(settings.VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._knowledge_base, f, ensure_ascii=False, indent=2)

    # --- متدهای مربوط به تسک ۱ و ۲ (تغییرات جزئی برای استفاده از متدهای جدید) ---

    def _generate_tags(self, user_query: str) -> list:
        """با استفاده از API OpenAI برای سوال کاربر تگ تولید میکند."""
        try:
            response = self._openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"""شما یک سیستم طبقهبندی دقیق هستید. یک  تگ مرتبط را از لیست زیر انتخاب کن. فقط و فقط یک آرایه JSON معتبر در خروجی برگردان. لیست تگهای مجاز: {json.dumps(settings.SUPPORT_TAGS, ensure_ascii=False)}"""
                    },
                    {
                        "role": "user",
                        "content": f"سوال: \"{user_query}\""
                    }
                ],
                temperature=0.0,
                max_tokens=100
            )
            # print("--- DEBUG RESPONSE ---")
            # print(f"نوع متغیر response: {type(response)}")
            # print(f"محتوای متغیر response: {response}")
            # print("--- END DEBUG ---")
            content = response.choices[0].message.content if response.choices and response.choices[0].message else ""
            tags = json.loads(content)

            if isinstance(tags, list):
                return tags
            elif isinstance(tags, dict) and "tags" in tags:
                return tags["tags"]
            else:
                return []


        except Exception :

            print(f"[OpenAI Tagging Error]: یک خطای غیرمنتظره رخ داد.")

            print("--- ردیابی کامل خطا (Traceback) ---")

            traceback.print_exc()

            print("---------------------------------")

            return []

    def _generate_response(self, user_query: str, context_docs: list) -> str:
        """با توجه به اسناد بازیابی شده و با استفاده از API OpenAI، پاسخ نهایی را تولید میکند."""

        if not context_docs:
            return "متاسفانه پاسخ مشخصی برای سوال شما در پایگاه دانش ما وجود ندارد."

        context_text = "\n\n---\n\n".join([
            f"سوال: {item['question']}\nپاسخ: {item['answer']}"
            for item in context_docs
        ])

        try:
            response = self._openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "شما یک کارشناس پشتیبانی حرفه‌ای و دقیق هستید. با توجه به اطلاعات زیر، به سوال کاربر به طور کامل و خلاصه پاسخ بده."
                    },
                    {
                        "role": "user",
                        "content": f"## اطلاعات مرتبط:\n{context_text}\n\n## سوال کاربر:\n\"{user_query}\""
                    }
                ],
                temperature=0.0,
                max_tokens=200
             )
            # print("--- DEBUG TAGS ---")
            # print(f"نوع متغیر response: {type(response)}")
            # print(f"محتوای متغیر response: {response}")
            # print("--- END DEBUG ---")
            choice = response.choices[0]
            content = getattr(choice.message, 'content', '') if choice.message else ''
            return content.strip()


        except Exception:

            print(f"[OpenAI Response Gen Error]: یک خطای غیرمنتظره رخ داد.")

            print("--- ردیابی کامل خطا (Traceback) ---")

            traceback.print_exc()

            print("---------------------------------")

            return "متاسفانه در ارتباط با سرویس OpenAI مشکلی پیش آمده است."

    def predict(self, user_query: str) -> dict:
        """
        نقطه ورود برای اندپوینت اصلی /qna. (با منطق جدید برای تسک ۴)
        """
        # اگر هیچ داده‌ای در پایگاه دانش وجود ندارد، مستقیم به اپراتور بفرست
        if len(self._knowledge_base) == 0:
            operator_tool(user_query)
            return {"tags_identified": [],
                    "final_answer": "متاسفانه پایگاه دانش خالی است. سوال شما برای بررسی توسط اپراتور ارسال شد.",
                    "retrieved_context_count": 0}

        # بخش ۱: ریترو کردن (بدون تغییر)
        user_vector = self._embedder.encode(user_query);
        similarities = cosine_similarity([user_vector], self._all_doc_vectors)[0];
        matches = sorted([(self._knowledge_base[i], sim) for i, sim in enumerate(similarities) if
                          sim >= settings.SIMILARITY_THRESHOLD], key=lambda x: x[1], reverse=True);
        top_matches = [m[0] for m in matches[:3]]

        # بخش ۲: تولید پاسخ (بدون تغییر)
        answer = self._generate_response(user_query, top_matches)
        tags = self._generate_tags(user_query)

        # === بخش جدید برای تسک ۴: حلقه بازخورد با اپراتور ===
        # اگر هیچ داکیومنتی پیدا نشد یا پاسخ تولید شده حاوی پیام پیش‌فرض "ندانستن" بود
        if not top_matches or "متاسفانه پاسخ مشخصی" in answer:
            operator_tool(user_query)

        return {"tags_identified": tags, "final_answer": answer, "retrieved_context_count": len(top_matches)}

    # === متدهای جدید برای تسک ۳ و ۴: مدیریت پایگاه دانش ===

    def _create_new_entry(self, qa_pair: dict) -> dict:
        """یک آیتم جدید با امبدینگ و ID ایجاد می‌کند."""
        text_to_embed = f"سوال: {qa_pair['question']}\n\nپاسخ: {qa_pair['answer']}"
        embedding = self._embedder.encode(text_to_embed, convert_to_numpy=True).tolist()

        # پیدا کردن بزرگترین ID عددی موجود برای ساخت ID جدید
        max_id = 0
        if self._knowledge_base:
            for item in self._knowledge_base:
                if item['id'].startswith('qna-'):
                    try:
                        num = int(item['id'].split('-')[1])
                        if num > max_id:
                            max_id = num
                    except (ValueError, IndexError):
                        continue

        return {
            "id": f"qna-{max_id + 1}",
            "question": qa_pair['question'],
            "answer": qa_pair['answer'],
            "embedding": embedding,
            "tags": []  # تگ‌ها می‌توانند بعدا اضافه شوند
        }

    def add_entries(self, qa_list: list) -> dict:
        """لیستی از پرسش و پاسخ‌ها را امبد کرده و به پایگاه دانش اضافه می‌کند."""
        new_items = [self._create_new_entry(qa) for qa in qa_list]
        self._knowledge_base.extend(new_items)
        self._save_knowledge_base()
        self._reload_knowledge_base()  # بارگذاری مجدد برای آپدیت بردارهای در حافظه
        return {"status": "success", "message": f"{len(new_items)} آیتم جدید اضافه شد."}

    def overwrite_database(self, qa_list: list) -> dict:
        """کل پایگاه دانش را با لیست جدیدی از پرسش و پاسخ‌ها جایگزین می‌کند."""
        self._knowledge_base = []  # پاک کردن کامل
        new_items = [self._create_new_entry(qa) for qa in qa_list]
        self._knowledge_base.extend(new_items)
        self._save_knowledge_base()
        self._reload_knowledge_base()
        return {"status": "success", "message": f"پایگاه دانش بازنویسی شد. تعداد آیتم‌های جدید: {len(new_items)}"}

    def delete_entries(self, ids_to_delete: list = None) -> dict:
        """تمام یا تعدادی از آیتم‌ها را از پایگاه دانش حذف می‌کند."""
        if ids_to_delete is None:  # اگر لیستی داده نشود، همه چیز را حذف کن
            self._knowledge_base = []
            message = "تمام آیتم‌ها از پایگاه دانش حذف شدند."
        else:  # در غیر این صورت، آیتم‌های مشخص شده را حذف کن
            initial_count = len(self._knowledge_base)
            self._knowledge_base = [item for item in self._knowledge_base if item.get('id') not in ids_to_delete]
            deleted_count = initial_count - len(self._knowledge_base)
            message = f"{deleted_count} آیتم با ID های مشخص شده حذف شدند."

        self._save_knowledge_base()
        self._reload_knowledge_base()
        return {"status": "success", "message": message}


# --- این بخش بدون تغییر باقی می‌ماند ---
model_instance = QAModel()


def get_model():
    return model_instance