# مسیر فایل: app/services/models.py
import uuid
import json
import os
import chromadb
import numpy as np
import requests
import traceback
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings
from app.services.tools import tools  # <-- وارد کردن ابزار جدید
import threading  # <-- برای جلوگیری از تداخل در زمان نوشتن فایل
from openai import OpenAI

class QAModel:
    """
    این کلاس، کل RAG سیستم به همراه منطق مدیریت پایگاه دانش و حلقه بازخورد را پیاده‌سازی می‌کند.
    """

    def __init__(self):
        self.COLLECTION_NAME = "knowledge_base_main"  # نام ثابت برای کالکشن
        self._embedder = None
        self._openai_client = None
        self._chroma_client = None
        self._collection = None
        self._load_dependencies()
        self._response_cache = {}
        self.CACHE_MAX_SIZE = 1000

    def _load_dependencies(self):
        """تمام نیازمندی‌های سنگین را فقط یک بار در زمان شروع برنامه بارگذاری می‌کند."""
        print("Service: در حال بارگذاری مدل امبدینگ...")
        self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

        print("Service: در حال مقداردهی اولیه کلاینت OpenAI...")
        self._openai_client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

        print("Service: در حال مقداردهی اولیه کلاینت ChromaDB...")
        # از PersistentClient برای ذخیره داده‌ها روی دیسک استفاده می‌کنیم
        self._chroma_client = chromadb.PersistentClient(path="./chroma_db_store")
        self._collection = self._chroma_client.get_or_create_collection(name=self.COLLECTION_NAME)

        print(f"Service: ChromaDB collection '{self.COLLECTION_NAME}' loaded. Total items: {self._collection.count()}")

    def _reload_knowledge_base(self):
        """پایگاه دانش را از فایل می‌خواند. این تابع باید هربار پس از تغییر فایل فراخوانی شود."""
        with self._file_lock:
            print("Service: در حال بارگذاری/بارگذاری مجدد پایگاه دانش برداری...")
            try:
                with open(settings.VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
                    self._knowledge_base = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # self._knowledge_base = []  # اگر فایل وجود نداشت یا خالی بود
                print("shit------------------------------------*")

        if self._knowledge_base:
            self._all_doc_vectors = np.array([item["embedding"] for item in self._knowledge_base])
        else:
            self._all_doc_vectors = np.array([])
        print(f"Service: items: {len(self._knowledge_base)}")
        print("VECTOR_STORE_PATH:", settings.VECTOR_STORE_PATH)
        print("File exists:", os.path.exists(settings.VECTOR_STORE_PATH))
        if os.path.exists(settings.VECTOR_STORE_PATH):
            with open(settings.VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
                raw = f.read()
            print("Raw file length:", len(raw))

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

        context_text = "\n\n---\n\n".join([f"سوال یافت شده: {item['question']}\nپاسخ مرتبط: {item['answer']}" for item in context_docs])
        print("=== DEBUG CONTEXT ===")
        print(context_text)

        try:
            response = self._openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful and friendly customer support agent. Your primary task is to answer the user's query based ONLY on the provided "context".

**CRITICAL INSTRUCTIONS:**
1.  **Language:** Your final answer **MUST** be in the **Persian (Farsi)** language. Do not write in English.
2.  **Source:** Use ONLY the information from the "context" section. Do not use your general knowledge.
3.  **Tone:** Your tone should be friendly, helpful, and natural. Avoid robotic and repetitive phrasing.
4.  **If Unsure:** If the answer is not found in the context, you MUST reply with the exact Persian phrase "پاسخ یافت نشد" and nothing else.
"""
                    },
                    {
                        "role": "user",
                        "content": f"## اطلاعات مرتبط:\n{context_text}\n\n## سوال کاربر:\n\"{user_query}\""
                    }
                ],
                temperature=0.5,
                top_p=0.9,
                max_tokens=100
             )
            #to do
            # crt output
            # print("--- DEBUG TAGS ---")
            # print(f"نوع متغیر response: {type(response)}")
            # print(f"محتوای متغیر response: {response}")
            # print("--- END DEBUG ---")
            choice = response.choices[0]
            content = getattr(choice.message, 'content', '') if choice.message else ''
            if "پاسخ یافت نشد" in content:
                return "متاسفانه پاسخ مشخصی برای سوال شما در پایگاه دانش ما وجود ندارد."
            return content.strip()


        except Exception:

            print(f"[OpenAI Response Gen Error]: یک خطای غیرمنتظره رخ داد.")

            print("--- ردیابی کامل خطا (Traceback) ---")

            traceback.print_exc()

            print("---------------------------------")

            return "متاسفانه در ارتباط با سرویس OpenAI مشکلی پیش آمده است."

    def _load_dependencies(self):
        """تمام نیازمندی‌های سنگین را فقط یک بار در زمان شروع برنامه بارگذاری می‌کند."""
        print("Service: در حال بارگذاری مدل امبدینگ...")
        self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

        print("Service: در حال مقداردهی اولیه کلاینت OpenAI...")
        self._openai_client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

        print("Service: در حال مقداردهی اولیه کلاینت ChromaDB...")
        # از PersistentClient برای ذخیره داده‌ها روی دیسک استفاده می‌کنیم
        self._chroma_client = chromadb.PersistentClient(path="./chroma_db_store")
        self._collection = self._chroma_client.get_or_create_collection(name=self.COLLECTION_NAME)

        print(f"Service: ChromaDB collection '{self.COLLECTION_NAME}' loaded. Total items: {self._collection.count()}")

    def predict(self, user_query: str) -> dict:
        if self._collection.count() == 0:
            ticket = tools.create_ticket(user_query)
            return {
                "status": "ticket_created",
                "message": "پایگاه دانش خالی است. سوال شما ثبت شد.",
                "question_id": ticket.question_id
            }

        results = self._collection.query(
            query_embeddings=[self._embedder.encode(user_query).tolist()],
            n_results=3
        )

        top_matches = []
        if results and results['ids'][0]:
            for i, distance in enumerate(results['distances'][0]):
                similarity = 1 - distance
                if similarity >= settings.SIMILARITY_THRESHOLD:
                    metadata = results['metadatas'][0][i]
                    top_matches.append({
                        "question": metadata.get('question'),
                        "answer": metadata.get('answer')
                    })

        if not top_matches:
            ticket = tools.create_ticket(user_query)
            return {
                "status": "ticket_created",
                "message": "پاسخ یافت نشد. سوال شما ثبت شد.",
                "question_id": ticket.question_id
            }

        answer = self._generate_response(user_query, top_matches)
        tags = self._generate_tags(user_query)

        response_data = {
            "final_answer": answer,
            "tags_identified": tags,
            "retrieved_context_count": len(top_matches),
            "original_question": user_query
        }

        session_id = str(uuid.uuid4())
        self._add_to_cache(session_id, response_data)

        response_data["session_id"] = session_id
        return response_data

    def handle_user_feedback(self, feedback_payload: dict):
        session_id = feedback_payload['session_id']
        is_correct = feedback_payload['is_correct']

        cached_response = self._response_cache.get(session_id)

        if not cached_response:
            return {"status": "error", "message": "شناسه جلسه نامعتبر است یا منقضی شده است."}

        if not is_correct:
            question = cached_response['original_question']
            incorrect_answer = cached_response['final_answer']

            tools.create_ticket(
                question=question,
                bot_answer=incorrect_answer,
                source="negative_feedback"
            )

            del self._response_cache[session_id]

            return {
                "status": "feedback_received",
                "message": "بازخورد شما ثبت شد. سوال برای بررسی توسط کارشناسان ما ارسال گردید."
            }

        question = cached_response['original_question']
        answer = cached_response['final_answer']

        print(f"INFO: بازخورد مثبت دریافت شد. در حال افزودن به پایگاه دانش: Q: '{question}'")

        self.add_entries([{"question": question, "answer": answer}])

        del self._response_cache[session_id]

        return {"status": "added_to_kb", "message": "متشکریم! پاسخ شما به بهبود دانش سیستم ما کمک کرد."}

    def _add_to_cache(self, session_id: str, response_data: dict):

        self._response_cache[session_id] = response_data


    def handle_admin_response(self, payload: dict):
        """منطق اصلی پردازش پاسخ ادمین."""
        question_id = payload['question_id']
        admin_answer = payload['answer']

        # 1. تیکت اصلی را پیدا کن تا متن سوال را به دست آوری
        ticket = tools.get_ticket(question_id)
        if not ticket:
            print(f"ERROR: تیکت با ID {question_id} یافت نشد.")
            return

        original_question = ticket['question']

        # 2. (شبیه‌سازی) پاسخ را مستقیماً برای کاربر ارسال کن
        print("=" * 50)
        print(f"SIMULATING: ارسال مستقیم پاسخ برای تیکت اصلی.")
        print(f"  > Ticket/User ID: {ticket.get('user_id', 'N/A')}")  # اگر user_id داشتید
        print(f"  > Original Question: {original_question}")
        print(f"  > Admin's Answer: {admin_answer}")
        print("=" * 50)
        # در اینجا کد واقعی ارسال ایمیل یا آپدیت API تیکتینگ قرار می‌گیرد

        # 3. پایگاه دانش را با سوال و جواب جدید آپدیت کن
        print("INFO: در حال افزودن سوال و جواب جدید به پایگاه دانش ChromaDB...")
        self.add_entries([{"question": original_question, "answer": admin_answer}])

        # 4. تیکت را از لیست انتظار حذف کن
        tools.close_ticket(question_id)


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
        """لیستی از پرسش و پاسخ‌ها را به ChromaDB اضافه می‌کند."""
        if not qa_list:
            return {"status": "noop", "message": "No items to add."}

        ids, documents, metadatas, embeddings = [], [], [], []
        max_id = self._get_max_qna_id()

        for i, qa_pair in enumerate(qa_list):
            new_id = f"qna-{max_id + i + 1}"
            text_to_embed = f"سوال: {qa_pair['question']}\n\nپاسخ: {qa_pair['answer']}"

            ids.append(new_id)
            documents.append(text_to_embed)
            embeddings.append(self._embedder.encode(text_to_embed).tolist())
            metadatas.append({"question": qa_pair['question'], "answer": qa_pair['answer']})

        self._collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return {"status": "success", "message": f"{len(ids)} آیتم جدید اضافه شد."}

    def overwrite_database(self, qa_list: list) -> dict:
        """کل پایگاه دانش را با لیست جدیدی از پرسش و پاسخ‌ها جایگزین می‌کند."""
        self._knowledge_base = []  # پاک کردن کامل
        new_items = [self._create_new_entry(qa) for qa in qa_list]
        self._knowledge_base.extend(new_items)
        self._save_knowledge_base()
        self._reload_knowledge_base()
        return {"status": "success", "message": f"پایگاه دانش بازنویسی شد. تعداد آیتم‌های جدید: {len(new_items)}"}

    def delete_entries(self, ids_to_delete: list = None) -> dict:
        """آیتم‌ها را از ChromaDB حذف می‌کند."""
        if ids_to_delete is None:
            count = self._collection.count()
            self._chroma_client.delete_collection(name=self.COLLECTION_NAME)
            self._collection = self._chroma_client.get_or_create_collection(name=self.COLLECTION_NAME)
            return {"status": "success", "message": f"کل پایگاه دانش ({count} آیتم) پاک شد."}

        self._collection.delete(ids=ids_to_delete)
        return {"status": "success", "message": f"{len(ids_to_delete)} آیتم حذف شدند."}

    def get_all_entries(self) -> list:
        """تمام آیتم‌ها را از ChromaDB بازیابی می‌کند."""
        results = self._collection.get(include=["metadatas"])
        if not results or not results['ids']: return []

        return [{"id": item_id, "question": meta.get("question"), "answer": meta.get("answer")}
                for item_id, meta in zip(results['ids'], results['metadatas'])]

    def _get_max_qna_id(self) -> int:
        """بزرگترین ID عددی را برای ساخت ID جدید پیدا می‌کند."""
        all_ids = self._collection.get(include=[])['ids']
        if not all_ids: return 0
        max_num = 0
        for item_id in all_ids:
            if item_id.startswith('qna-'):
                try:
                    num = int(item_id.split('-')[1])
                    if num > max_num: max_num = num
                except (ValueError, IndexError):
                    continue
        return max_num


# --- این بخش بدون تغییر باقی می‌ماند ---
model_instance = QAModel()


def get_model():
    return model_instance