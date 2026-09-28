# مسیر فایل: app/services/models/agent.py
import uuid
import json
import threading
import traceback
from collections import OrderedDict

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from app.core.config import settings
from app.services.tools import tools

NOT_FOUND_MESSAGE = "متاسفانه پاسخ مشخصی برای سوال شما در پایگاه دانش ما وجود ندارد."


class QAModel:
    """
    این کلاس، کل RAG سیستم به همراه منطق مدیریت پایگاه دانش و حلقه بازخورد را پیاده‌سازی می‌کند.
    """

    COLLECTION_NAME = "knowledge_base_main"  # نام ثابت برای کالکشن

    def __init__(self):
        self._embedder = None
        self._openai_client = None
        self._chroma_client = None
        self._collection = None
        # قفل برای عملیات نوشتن روی کالکشن (ساخت ID جدید و بازسازی کالکشن)
        self._kb_lock = threading.RLock()
        # کش پاسخ‌ها برای حلقه بازخورد (LRU با سقف مشخص)
        self._response_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self._load_dependencies()

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
        self._chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self._collection = self._get_or_create_collection()

        print(f"Service: ChromaDB collection '{self.COLLECTION_NAME}' loaded. Total items: {self._collection.count()}")

    def _get_or_create_collection(self):
        # کالکشن‌های جدید با فاصله کسینوسی ساخته می‌شوند؛ کالکشن‌های قدیمی فضای فعلی خود را حفظ می‌کنند.
        return self._chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, text: str) -> list:
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

    def _distance_to_similarity(self, distance: float) -> float:
        """فاصله برگشتی ChromaDB را به شباهت کسینوسی تبدیل می‌کند."""
        space = (self._collection.metadata or {}).get("hnsw:space", "l2")
        if space == "l2":
            # برای بردارهای نرمال‌شده: فاصله L2 مربعی = 2 - 2cos
            return 1 - distance / 2
        if space == "ip":
            return 1 - distance
        return 1 - distance  # cosine

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
            content = response.choices[0].message.content if response.choices and response.choices[0].message else ""
            content = (content or "").strip()
            # حذف ```json ... ``` در صورت وجود
            if content.startswith("```"):
                content = content.strip("`").removeprefix("json").strip()
            tags = json.loads(content)

            if isinstance(tags, dict):
                tags = tags.get("tags", [])
            if not isinstance(tags, list):
                return []
            # فقط تگ‌های مجاز را نگه دار
            return [tag for tag in tags if tag in settings.SUPPORT_TAGS]

        except Exception:
            print("[OpenAI Tagging Error]: یک خطای غیرمنتظره رخ داد.")
            traceback.print_exc()
            return []

    def _generate_response(self, user_query: str, context_docs: list) -> str:
        """با توجه به اسناد بازیابی شده و با استفاده از API OpenAI، پاسخ نهایی را تولید میکند."""

        if not context_docs:
            return NOT_FOUND_MESSAGE

        context_text = "\n\n---\n\n".join([f"سوال یافت شده: {item['question']}\nپاسخ مرتبط: {item['answer']}" for item in context_docs])

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
                max_tokens=settings.LLM_MAX_TOKENS
            )
            choice = response.choices[0]
            content = (getattr(choice.message, 'content', '') if choice.message else '') or ''
            if "پاسخ یافت نشد" in content:
                return NOT_FOUND_MESSAGE
            return content.strip()

        except Exception:
            print("[OpenAI Response Gen Error]: یک خطای غیرمنتظره رخ داد.")
            traceback.print_exc()
            return "متاسفانه در ارتباط با سرویس OpenAI مشکلی پیش آمده است."

    def predict(self, user_query: str) -> dict:
        if self._collection.count() == 0:
            ticket = tools.create_ticket(user_query)
            return {
                "status": "ticket_created",
                "message": "پایگاه دانش خالی است. سوال شما ثبت شد.",
                "question_id": ticket.question_id
            }

        results = self._collection.query(
            query_embeddings=[self._embed(user_query)],
            n_results=settings.TOP_K,
            include=["metadatas", "distances"],
        )

        top_matches = []
        if results and results['ids'] and results['ids'][0]:
            for i, distance in enumerate(results['distances'][0]):
                similarity = self._distance_to_similarity(distance)
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

        cached_response = self._pop_from_cache(session_id)

        if not cached_response:
            return {"status": "error", "message": "شناسه جلسه نامعتبر است یا منقضی شده است."}

        question = cached_response['original_question']
        answer = cached_response['final_answer']

        if not is_correct:
            tools.create_ticket(
                question=question,
                bot_answer=answer,
                source="negative_feedback"
            )
            return {
                "status": "feedback_received",
                "message": "بازخورد شما ثبت شد. سوال برای بررسی توسط کارشناسان ما ارسال گردید."
            }

        print(f"INFO: بازخورد مثبت دریافت شد. در حال افزودن به پایگاه دانش: Q: '{question}'")
        self.add_entries([{"question": question, "answer": answer}])

        return {"status": "added_to_kb", "message": "متشکریم! پاسخ شما به بهبود دانش سیستم ما کمک کرد."}

    def _add_to_cache(self, session_id: str, response_data: dict):
        with self._cache_lock:
            self._response_cache[session_id] = response_data
            self._response_cache.move_to_end(session_id)
            while len(self._response_cache) > settings.CACHE_MAX_SIZE:
                self._response_cache.popitem(last=False)

    def _pop_from_cache(self, session_id: str):
        # pop اتمی است تا یک session_id فقط یک بار بازخورد بگیرد
        with self._cache_lock:
            return self._response_cache.pop(session_id, None)

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
        print("SIMULATING: ارسال مستقیم پاسخ برای تیکت اصلی.")
        print(f"  > Ticket/User ID: {ticket.get('user_id', 'N/A')}")
        print(f"  > Original Question: {original_question}")
        print(f"  > Admin's Answer: {admin_answer}")
        print("=" * 50)
        # در اینجا کد واقعی ارسال ایمیل یا آپدیت API تیکتینگ قرار می‌گیرد

        # 3. پایگاه دانش را با سوال و جواب جدید آپدیت کن
        print("INFO: در حال افزودن سوال و جواب جدید به پایگاه دانش ChromaDB...")
        self.add_entries([{"question": original_question, "answer": admin_answer}])

        # 4. تیکت را از لیست انتظار حذف کن
        tools.close_ticket(question_id)

    def add_entries(self, qa_list: list) -> dict:
        """لیستی از پرسش و پاسخ‌ها را به ChromaDB اضافه می‌کند."""
        if not qa_list:
            return {"status": "noop", "message": "No items to add."}

        with self._kb_lock:
            ids, documents, metadatas, embeddings = [], [], [], []
            max_id = self._get_max_qna_id()

            for i, qa_pair in enumerate(qa_list):
                new_id = f"qna-{max_id + i + 1}"
                text_to_embed = f"سوال: {qa_pair['question']}\n\nپاسخ: {qa_pair['answer']}"

                ids.append(new_id)
                documents.append(text_to_embed)
                embeddings.append(self._embed(text_to_embed))
                metadatas.append({"question": qa_pair['question'], "answer": qa_pair['answer']})

            self._collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return {"status": "success", "message": f"{len(ids)} آیتم جدید اضافه شد."}

    def _reset_collection(self) -> int:
        """کالکشن را کامل پاک و دوباره (با فاصله کسینوسی) می‌سازد. تعداد آیتم‌های حذف‌شده را برمی‌گرداند."""
        count = self._collection.count()
        self._chroma_client.delete_collection(name=self.COLLECTION_NAME)
        self._collection = self._get_or_create_collection()
        return count

    def overwrite_database(self, qa_list: list) -> dict:
        """کل پایگاه دانش را با لیست جدیدی از پرسش و پاسخ‌ها جایگزین می‌کند."""
        with self._kb_lock:
            self._reset_collection()
            self.add_entries(qa_list)
        return {"status": "success", "message": f"پایگاه دانش بازنویسی شد. تعداد آیتم‌های جدید: {len(qa_list)}"}

    def delete_entries(self, ids_to_delete: list = None) -> dict:
        """آیتم‌ها را از ChromaDB حذف می‌کند."""
        with self._kb_lock:
            if ids_to_delete is None:
                count = self._reset_collection()
                return {"status": "success", "message": f"کل پایگاه دانش ({count} آیتم) پاک شد."}
            if not ids_to_delete:
                return {"status": "noop", "message": "No items to delete."}

            self._collection.delete(ids=ids_to_delete)
        return {"status": "success", "message": f"{len(ids_to_delete)} آیتم حذف شدند."}

    def get_all_entries(self) -> list:
        """تمام آیتم‌ها را از ChromaDB بازیابی می‌کند."""
        results = self._collection.get(include=["metadatas"])
        if not results or not results['ids']:
            return []

        return [{"id": item_id, "question": meta.get("question"), "answer": meta.get("answer")}
                for item_id, meta in zip(results['ids'], results['metadatas'])]

    def _get_max_qna_id(self) -> int:
        """بزرگترین ID عددی را برای ساخت ID جدید پیدا می‌کند."""
        all_ids = self._collection.get(include=[])['ids']
        max_num = 0
        for item_id in all_ids:
            if item_id.startswith('qna-'):
                try:
                    num = int(item_id.split('-')[1])
                    if num > max_num:
                        max_num = num
                except (ValueError, IndexError):
                    continue
        return max_num


model_instance = QAModel()


def get_model():
    return model_instance
