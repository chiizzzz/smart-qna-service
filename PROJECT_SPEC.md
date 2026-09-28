# مشخصات کامل پروژه Smart Q&A Service

> این سند طوری نوشته شده که یک ایجنت یا برنامه‌نویس **بدون دیدن کد** بتواند منطق، ساختار، رفتار دقیق و محدودیت‌های سیستم را بفهمد و در صورت نیاز آن را **عیناً بازسازی** کند. نام فایل‌ها، کلاس‌ها، توابع، فیلدها و پیام‌ها دقیقاً همان چیزی است که در کد وجود دارد.

---

## ۱. خلاصه در یک پاراگراف

یک سرویس وب **FastAPI** (پایتون) برای پشتیبانی مشتری فارسی‌زبان. کاربر سوال می‌پرسد؛ سوال با مدل **Sentence Transformers** به بردار تبدیل می‌شود و در پایگاه دانش برداری **ChromaDB** (ذخیره روی دیسک) جستجو می‌شود. اگر موارد مشابه با شباهت کافی پیدا شد، یک **LLM از طریق API سازگار با OpenAI** فقط بر اساس همان موارد یک پاسخ فارسی می‌سازد و سوال را در یکی از ۷ دسته پشتیبانی **تگ** می‌زند. اگر چیزی پیدا نشد، برای سوال **تیکت** در یک فایل JSON ثبت می‌شود. کاربر می‌تواند روی پاسخ **بازخورد** بدهد: بازخورد مثبت ← جفت سوال/پاسخ به پایگاه دانش اضافه می‌شود؛ بازخورد منفی ← تیکت ساخته می‌شود. **ادمین** (با توکن) می‌تواند پایگاه دانش را مدیریت کند و به تیکت‌ها پاسخ دهد؛ پاسخ ادمین به پایگاه دانش اضافه و تیکت بسته می‌شود. به این ترتیب سیستم یک **حلقه یادگیری مداوم** دارد.

---

## ۲. تکنولوژی‌ها و وابستگی‌ها

| بخش | ابزار | نسخه در `requirements.txt` |
|---|---|---|
| وب فریم‌ورک | FastAPI + Uvicorn | `fastapi==0.116.1`, `uvicorn==0.35.0` |
| اعتبارسنجی و تنظیمات | Pydantic v2 + pydantic-settings + python-dotenv | `pydantic==2.11.7`, `pydantic-settings==2.10.1`, `python-dotenv==1.1.1` |
| LLM | کتابخانه `openai` (کلاس `OpenAI`) | `openai==1.98.0` |
| پایگاه دانش برداری | ChromaDB (`PersistentClient`) | `chromadb==1.5.9` |
| امبدینگ | `sentence-transformers` (+ `torch`) | `sentence-transformers==5.0.0`, `torch==2.7.1` |

پایتون ۳.۹ به بالا (از `str.removeprefix` استفاده شده).

---

## ۳. ساختار فایل‌ها

```
smart-qna-service/
├── app/
│   ├── __init__.py                 (خالی)
│   ├── main.py                     ساخت اپ FastAPI، مسیرهای / و /ping
│   ├── schemas.py                  همه مدل‌های Pydantic
│   ├── core/
│   │   ├── __init__.py             (خالی)
│   │   └── config.py               کلاس Settings و شیء سراسری settings
│   ├── api/
│   │   ├── __init__.py             (خالی)
│   │   └── routes.py               همه endpointها + احراز هویت ادمین
│   └── services/
│       ├── __init__.py             (خالی)
│       ├── models/
│       │   ├── __init__.py         (خالی)
│       │   └── agent.py            کلاس QAModel (هسته RAG) + singleton
│       └── tools/
│           ├── __init__.py         (خالی)
│           └── tools.py            مدیریت تیکت‌ها در فایل JSON
├── scripts/
│   ├── __init__.py                 (خالی)
│   └── import_knowledge_base.py    وارد کردن دسته‌ای Q&A از فایل JSON
├── New folder/app.zip              نسخه قدیمی پروژه + vector_store.json (۱۱۵ Q&A)؛ فقط آرشیو
├── requirements.txt
├── .env.example                    نمونه متغیرهای محیطی
├── .gitignore
├── README.md / README(farsi).md
└── LICENSE
```

فایل‌ها/پوشه‌هایی که در **زمان اجرا** ساخته می‌شوند و در گیت نیستند:
- `./chroma_db_store/` — داده ChromaDB (مسیر قابل تنظیم با `CHROMA_DB_PATH`)
- `./pending_tickets.json` — تیکت‌ها (مسیر ثابت، نسبت به پوشه‌ای که سرور از آن اجرا می‌شود)
- `.env` — تنظیمات محرمانه

---

## ۴. تنظیمات (`app/core/config.py`)

کلاس `Settings(BaseSettings)` با `model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")`. مقادیر از متغیرهای محیطی یا فایل `.env` خوانده می‌شوند. یک شیء سراسری `settings = Settings()` در زمان import ساخته می‌شود؛ **اگر `OPENAI_API_KEY` تعریف نشده باشد، برنامه همان لحظه با خطای اعتبارسنجی Pydantic بالا نمی‌آید.**

| نام | نوع | پیش‌فرض | کاربرد |
|---|---|---|---|
| `API_PREFIX` | str | `"/api/v1"` | پیشوند همه مسیرهای router |
| `OPENAI_API_KEY` | str | **اجباری** | کلید API مدل زبانی |
| `OPENAI_BASE_URL` | Optional[str] | `None` | آدرس API سازگار با OpenAI؛ `None` یعنی api.openai.com |
| `LLM_MODEL` | str | `"gpt-4o"` | نام مدل برای تولید پاسخ و تگ |
| `LLM_MAX_TOKENS` | int | `512` | سقف توکن پاسخ نهایی (نه تگ‌گذاری) |
| `EMBEDDING_MODEL` | str | `"intfloat/multilingual-e5-large"` | مدل Sentence Transformers |
| `CHROMA_DB_PATH` | str | `"./chroma_db_store"` | مسیر ذخیره ChromaDB |
| `SIMILARITY_THRESHOLD` | float | `0.75` | حداقل شباهت کسینوسی برای پذیرفتن یک نتیجه |
| `TOP_K` | int | `3` | تعداد نتایج بازیابی از ChromaDB |
| `CACHE_MAX_SIZE` | int | `1000` | سقف کش پاسخ‌ها برای بازخورد |
| `ADMIN_API_KEY` | Optional[str] | `None` | توکن ادمین؛ خالی = مسیرهای ادمین غیرفعال (503) |
| `LANGFUSE_PUBLIC_KEY` | Optional[str] | `None` | تعریف شده ولی **در هیچ جای کد استفاده نمی‌شود** |
| `LANGFUSE_SECRET_KEY` | Optional[str] | `None` | همان |
| `SUPPORT_TAGS` | List[str] | ۷ تگ زیر | دسته‌های مجاز تگ‌گذاری |

`SUPPORT_TAGS` به ترتیب:
1. `پشتیبانی فنی`
2. `فروش و قیمت‌گذاری`
3. `مالی و صورتحساب`
4. `حساب کاربری و ورود`
5. `ارسال و تحویل`
6. `پیشنهادات و انتقادات`
7. `همکاری تجاری`

---

## ۵. راه‌اندازی برنامه (`app/main.py`) و ترتیب بارگذاری

1. `app = FastAPI(title="Smart Q&A Service")`.
2. `print("✅ API_PREFIX =", settings.API_PREFIX)`.
3. `app.include_router(routes.router, prefix=settings.API_PREFIX)`.
4. **مهم:** import کردن `app.api.routes` باعث import شدن `app.services.models.agent` می‌شود و در انتهای آن ماژول `model_instance = QAModel()` اجرا می‌شود. یعنی **مدل امبدینگ، کلاینت OpenAI و ChromaDB در زمان import بارگذاری می‌شوند** (نه در startup event). بارگذاری اول ممکن است دانلود مدل از HuggingFace را هم شامل شود.
5. مسیرهای خود `main.py`:
   - `GET /` ← `{"status": "ok", "message": "Welcome to the Smart Q&A "}` (با فاصله در انتها)
   - `GET /ping` ← چاپ `📡 [MAIN] پینگ از route اصلی` و برگرداندن `{"status": "pong"}`
6. رویداد `@app.on_event("startup")` به نام `show_all_routes`: همه مسیرها را به شکل `🔗 {path} → {name}` چاپ می‌کند.

اجرا: `uvicorn app.main:app --host 0.0.0.0 --port 8000` از ریشه پروژه.

---

## ۶. مدل‌های داده (`app/schemas.py`)

همه `pydantic.BaseModel` هستند.

| مدل | فیلدها | کاربرد |
|---|---|---|
| `QARequest` | `query: str` | بدنه `POST /qna` |
| `QAResponse` | `session_id: str` (پیش‌فرض uuid4)، `tags_identified: List[str]`، `final_answer: str`، `retrieved_context_count: int`، `original_question: str` | پاسخ موفق `/qna` |
| `TicketCreationResponse` | `status: Literal["ticket_created"] = "ticket_created"`، `message: str`، `question_id: str` | پاسخ `/qna` وقتی تیکت ساخته شد |
| `FeedbackRequest` | `session_id: str`، `is_correct: bool` (اجباری) | بدنه `/feedback` |
| `FeedbackResponse` | `status: str`، `message: str` | پاسخ `/feedback` |
| `SingleQA` | `question: str`، `answer: str` | یک جفت Q&A |
| `QAList` | `data: List[SingleQA]` | بدنه add و overwrite |
| `OperatorAnswerPayload` | `question: str`، `answer: str` | بدنه add_from_operator |
| `DeleteRequest` | `ids: Optional[List[str]] = None` | بدنه delete |
| `KnowledgeBaseItem` | `id: str`، `question: str`، `answer: str` | یک آیتم پایگاه دانش |
| `KnowledgeBaseDump` | `total_items: int`، `data: List[KnowledgeBaseItem]` | پاسخ `/knowledge_base/all` |
| `PendingTicket` | `question_id: str` (پیش‌فرض uuid4)، `question: str`، `timestamp: datetime` (پیش‌فرض `datetime.now`، بدون timezone)، `bot_answer: Optional[str] = None`، `ticket_source: Literal["unanswered", "negative_feedback"] = "unanswered"` | یک تیکت |
| `PendingTicketList` | `data: List[PendingTicket]` | پاسخ `/pending_tickets` |
| `AdminTicketResponsePayload` | `question_id: str`، `answer: str` | بدنه respond_to_ticket |
| `AddKnowledgeResponse` | `message`، `items_added_count`، `total_items_in_db` | **تعریف شده ولی استفاده نمی‌شود** |

---

## ۷. API کامل (`app/api/routes.py`)

دو router وجود دارد: `router` (عمومی) و `admin_router` که با `prefix="/admin"` و تگ `"Admin: Knowledge Base & Ticketing"` داخل `router` قرار می‌گیرد. کل `router` با پیشوند `/api/v1` به اپ وصل است. همه endpointها **همگام (sync `def`)** هستند؛ FastAPI آن‌ها را در threadpool اجرا می‌کند، پس چند درخواست ممکن است هم‌زمان روی یک `QAModel` کار کنند (به همین دلیل قفل‌ها وجود دارند). مدل با `Depends(get_model)` تزریق می‌شود که همیشه همان singleton را برمی‌گرداند.

### ۷.۱ احراز هویت ادمین

تابع `require_admin(x_admin_token: Optional[str] = Header(default=None))` به‌صورت dependency روی کل `admin_router` اعمال شده (یعنی هدر HTTP به نام `X-Admin-Token`):
1. اگر `settings.ADMIN_API_KEY` خالی/None باشد ← `503` با detail: `ADMIN_API_KEY تنظیم نشده است؛ مسیرهای ادمین غیرفعال هستند.`
2. اگر هدر نباشد یا با `secrets.compare_digest` برابر نباشد ← `401` با detail: `توکن ادمین نامعتبر است.`
3. در غیر این صورت عبور.

مسیرهای عمومی (`/qna`، `/feedback`، `/`، `/ping`) هیچ احراز هویتی ندارند.

### ۷.۲ جدول endpointها

| متد | مسیر کامل | بدنه | کد موفق | کاری که انجام می‌دهد |
|---|---|---|---|---|
| GET | `/` | — | 200 | خوش‌آمد |
| GET | `/ping` | — | 200 | `{"status":"pong"}` |
| POST | `/api/v1/qna` | `QARequest` | 200 | `model.predict(query)`؛ اگر نتیجه `status == "ticket_created"` داشت ← `TicketCreationResponse`، وگرنه `QAResponse` |
| POST | `/api/v1/feedback` | `FeedbackRequest` | 200 | `model.handle_user_feedback(payload.model_dump())` ← `FeedbackResponse` |
| POST | `/api/v1/admin/knowledge_base/add` | `QAList` | 201 | `model.add_entries(data)` |
| PUT | `/api/v1/admin/knowledge_base/overwrite` | `QAList` | 200 | `model.overwrite_database(data)` |
| DELETE | `/api/v1/admin/knowledge_base/delete` | `DeleteRequest` یا بدون بدنه | 200 | `model.delete_entries(ids)`؛ بدون بدنه یا `ids` خالی/None ← **پاک شدن کل پایگاه دانش** |
| POST | `/api/v1/admin/knowledge_base/add_from_operator` | `OperatorAnswerPayload` | 201 | `model.add_entries([payload])` |
| GET | `/api/v1/admin/knowledge_base/all` | — | 200 | `KnowledgeBaseDump` از `model.get_all_entries()` |
| GET | `/api/v1/admin/pending_tickets` | — | 200 | `PendingTicketList(data=tools.list_tickets())` |
| POST | `/api/v1/admin/respond_to_ticket` | `AdminTicketResponsePayload` | 202 | اگر `tools.get_ticket(question_id)` وجود نداشت ← `404` با detail `تیکتی با این شناسه یافت نشد.`؛ وگرنه `model.handle_admin_response(payload)` را به **BackgroundTasks** می‌دهد و فوراً `{"message": "پاسخ شما ثبت شد و پردازش آن در پس‌زمینه انجام می‌شود."}` برمی‌گرداند |

نکته: با `ids` خالی (`{"ids": []}`) مقدار `[]` به `delete_entries` می‌رسد که `None` نیست، پس `{"status": "noop", "message": "No items to delete."}` برمی‌گردد و کل پایگاه پاک **نمی‌شود**. فقط `ids` برابر `None` (بدنه خالی، `{}`، یا `{"ids": null}`) کل پایگاه را پاک می‌کند.

خطاهای اعتبارسنجی بدنه طبق رفتار استاندارد FastAPI کد `422` برمی‌گردانند.

---

## ۸. هسته سیستم: کلاس `QAModel` (`app/services/models/agent.py`)

### ۸.۱ ثابت‌ها و وضعیت داخلی

- ثابت ماژول: `NOT_FOUND_MESSAGE = "متاسفانه پاسخ مشخصی برای سوال شما در پایگاه دانش ما وجود ندارد."`
- ثابت کلاس: `COLLECTION_NAME = "knowledge_base_main"`
- فیلدها:
  - `_embedder`: شیء `SentenceTransformer(settings.EMBEDDING_MODEL)`
  - `_openai_client`: `OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)`
  - `_chroma_client`: `chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)`
  - `_collection`: کالکشن ChromaDB
  - `_kb_lock`: `threading.RLock()` — روی همه عملیات نوشتن پایگاه دانش (reentrant چون `overwrite_database` داخل خودش `add_entries` را صدا می‌زند)
  - `_response_cache`: `OrderedDict` — کش LRU پاسخ‌ها با کلید `session_id`
  - `_cache_lock`: `threading.Lock()` — روی کش
- در انتهای ماژول: `model_instance = QAModel()` و `def get_model(): return model_instance` (singleton سراسری).

### ۸.۲ `_load_dependencies()`
به ترتیب: بارگذاری مدل امبدینگ، ساخت کلاینت OpenAI، ساخت `PersistentClient`، گرفتن/ساختن کالکشن با `_get_or_create_collection()`، و چاپ تعداد آیتم‌ها. پیام‌ها با پیشوند `Service:` چاپ می‌شوند.

### ۸.۳ `_get_or_create_collection()`
`get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})`.
- کالکشن **جدید** با فاصله کسینوسی ساخته می‌شود.
- اگر کالکشن از قبل وجود داشته باشد، ChromaDB metadata فضای فاصله را **تغییر نمی‌دهد**؛ کالکشن‌های قدیمی (ساخته‌شده قبل از این تغییر) ممکن است فضای پیش‌فرض `l2` داشته باشند.

### ۸.۴ `_embed(text) -> list[float]`
`self._embedder.encode(text, normalize_embeddings=True).tolist()` — همیشه بردار **نرمال‌شده** (طول ۱).

### ۸.۵ `_distance_to_similarity(distance) -> float`
فضا از `self._collection.metadata.get("hnsw:space", "l2")` خوانده می‌شود:
- `l2`: ChromaDB مربع فاصله اقلیدسی برمی‌گرداند؛ برای بردار نرمال `d = 2 - 2cos` پس **`similarity = 1 - d/2`**
- `ip`: `similarity = 1 - d`
- `cosine`: ChromaDB `d = 1 - cos` برمی‌گرداند پس **`similarity = 1 - d`**

خروجی همیشه یک شباهت کسینوسی بین −۱ و ۱ است و با `SIMILARITY_THRESHOLD` مقایسه می‌شود.

### ۸.۶ ساختار هر آیتم در ChromaDB

| بخش | مقدار |
|---|---|
| `id` | `"qna-{n}"` (عدد صحیح مثبت) |
| `document` | `"سوال: {question}\n\nپاسخ: {answer}"` |
| `embedding` | `_embed(document)` — یعنی امبدینگ **ترکیب سوال و پاسخ** |
| `metadata` | `{"question": question, "answer": answer}` |

توجه: در جستجو فقط **متن خام سوال کاربر** امبد می‌شود ولی آیتم‌ها از «سوال + پاسخ» امبد شده‌اند (عدم تقارن). پیشوندهای `query:`/`passage:` که مدل‌های E5 توصیه می‌کنند استفاده **نمی‌شوند**.

### ۸.۷ `predict(user_query) -> dict` — الگوریتم اصلی پرسش و پاسخ

```
1. اگر collection.count() == 0:
       ticket = tools.create_ticket(user_query)        # source="unanswered"
       return {"status": "ticket_created",
               "message": "پایگاه دانش خالی است. سوال شما ثبت شد.",
               "question_id": ticket.question_id}

2. results = collection.query(query_embeddings=[_embed(user_query)],
                              n_results=settings.TOP_K,
                              include=["metadatas", "distances"])

3. top_matches = []
   برای هر نتیجه i (به ترتیب نزدیک‌ترین):
       sim = _distance_to_similarity(distance_i)
       اگر sim >= SIMILARITY_THRESHOLD:
           top_matches.append({"question": meta_i["question"], "answer": meta_i["answer"]})

4. اگر top_matches خالی بود:
       ticket = tools.create_ticket(user_query)
       return {"status": "ticket_created",
               "message": "پاسخ یافت نشد. سوال شما ثبت شد.",
               "question_id": ticket.question_id}

5. answer = _generate_response(user_query, top_matches)     # فراخوانی LLM اول
6. tags   = _generate_tags(user_query)                       # فراخوانی LLM دوم (پشت سر هم، نه موازی)

7. response_data = {"final_answer": answer, "tags_identified": tags,
                    "retrieved_context_count": len(top_matches),
                    "original_question": user_query}
8. session_id = uuid4 ؛ _add_to_cache(session_id, response_data)
9. response_data["session_id"] = session_id ؛ return response_data
```

نکات رفتاری مهم:
- اگر LLM بگوید «پاسخ یافت نشد» یا خطا بدهد، باز هم **پاسخ عادی (`QAResponse`) برمی‌گردد** با `final_answer` برابر پیام «یافت نشد» یا پیام خطا — **تیکت ساخته نمی‌شود**. تیکت فقط در مراحل ۱ و ۴ ساخته می‌شود.
- `retrieved_context_count` تعداد مواردی است که از آستانه رد شده‌اند (۱ تا `TOP_K`).
- چون شیء cache و شیء بازگشتی یک دیکشنری هستند، `session_id` پس از کش شدن به همان دیکشنری اضافه می‌شود (بی‌ضرر).

### ۸.۸ `_generate_response(user_query, context_docs) -> str`

- اگر `context_docs` خالی ← `NOT_FOUND_MESSAGE` (عملاً رخ نمی‌دهد چون `predict` قبلش چک می‌کند).
- متن زمینه: هر آیتم به شکل `"سوال یافت شده: {question}\nپاسخ مرتبط: {answer}"` و بین آیتم‌ها `"\n\n---\n\n"`.
- فراخوانی `chat.completions.create` با `model=LLM_MODEL`، `temperature=0.5`، `top_p=0.9`، `max_tokens=LLM_MAX_TOKENS`.
- پیام system (انگلیسی، عیناً):

```
You are a helpful and friendly customer support agent. Your primary task is to answer the user's query based ONLY on the provided "context".

**CRITICAL INSTRUCTIONS:**
1.  **Language:** Your final answer **MUST** be in the **Persian (Farsi)** language. Do not write in English.
2.  **Source:** Use ONLY the information from the "context" section. Do not use your general knowledge.
3.  **Tone:** Your tone should be friendly, helpful, and natural. Avoid robotic and repetitive phrasing.
4.  **If Unsure:** If the answer is not found in the context, you MUST reply with the exact Persian phrase "پاسخ یافت نشد" and nothing else.
```

- پیام user: `"## اطلاعات مرتبط:\n{context_text}\n\n## سوال کاربر:\n\"{user_query}\""`
- پس‌پردازش: اگر متن خروجی **شامل** `پاسخ یافت نشد` بود ← `NOT_FOUND_MESSAGE`؛ وگرنه `content.strip()`.
- هر Exception ← چاپ traceback و برگرداندن `"متاسفانه در ارتباط با سرویس OpenAI مشکلی پیش آمده است."`

### ۸.۹ `_generate_tags(user_query) -> list[str]`

- فراخوانی `chat.completions.create` با `temperature=0.0`، `max_tokens=100`.
- پیام system (عیناً، شامل غلط‌های تایپی اصلی):
  `شما یک سیستم طبقهبندی دقیق هستید. یک  تگ مرتبط را از لیست زیر انتخاب کن. فقط و فقط یک آرایه JSON معتبر در خروجی برگردان. لیست تگهای مجاز: {SUPPORT_TAGS به صورت JSON با ensure_ascii=False}`
- پیام user: `سوال: "{user_query}"`
- پس‌پردازش:
  1. `strip()`؛ اگر با ```` ``` ```` شروع شد، بک‌تیک‌ها از دو طرف حذف، پیشوند `json` حذف و دوباره `strip()`.
  2. `json.loads`.
  3. اگر dict بود ← مقدار کلید `"tags"` (پیش‌فرض `[]`).
  4. اگر list نبود ← `[]`.
  5. فقط عناصری که **دقیقاً** در `SUPPORT_TAGS` هستند نگه داشته می‌شوند.
- هر Exception (از جمله JSON نامعتبر) ← چاپ traceback و `[]`.

### ۸.۱۰ کش پاسخ‌ها

- `_add_to_cache(session_id, data)`: زیر `_cache_lock` اضافه می‌کند، به انتها منتقل می‌کند، و تا زمانی که اندازه از `CACHE_MAX_SIZE` بیشتر است قدیمی‌ترین را حذف می‌کند (LRU/FIFO).
- `_pop_from_cache(session_id)`: زیر قفل `pop` می‌کند (اتمی) — یعنی **هر session_id فقط یک بار** قابل بازخورد است.
- کش **فقط در حافظه** است: با ری‌استارت سرور یا اجرای چند worker/پروسه، `session_id`ها گم می‌شوند یا بین پروسه‌ها مشترک نیستند.

### ۸.۱۱ `handle_user_feedback(payload) -> dict`

```
cached = _pop_from_cache(session_id)
اگر cached نبود ← {"status": "error", "message": "شناسه جلسه نامعتبر است یا منقضی شده است."}

question = cached["original_question"] ; answer = cached["final_answer"]

اگر is_correct == False:
    tools.create_ticket(question=question, bot_answer=answer, source="negative_feedback")
    return {"status": "feedback_received",
            "message": "بازخورد شما ثبت شد. سوال برای بررسی توسط کارشناسان ما ارسال گردید."}

اگر is_correct == True:
    add_entries([{"question": question, "answer": answer}])
    return {"status": "added_to_kb",
            "message": "متشکریم! پاسخ شما به بهبود دانش سیستم ما کمک کرد."}
```

توجه: وضعیت خطا هم با HTTP 200 برمی‌گردد (فقط `status: "error"` در بدنه). بازخورد مثبت هر `final_answer`ی را اضافه می‌کند، **حتی اگر آن پاسخ پیام «یافت نشد» یا پیام خطای OpenAI باشد** (محدودیت شناخته‌شده).

### ۸.۱۲ `handle_admin_response(payload)` (در پس‌زمینه اجرا می‌شود)

1. `ticket = tools.get_ticket(question_id)`؛ اگر نبود چاپ `ERROR: تیکت با ID ... یافت نشد.` و return (ممکن است بین چک route و اجرای background تیکت حذف شده باشد).
2. **شبیه‌سازی** ارسال پاسخ به کاربر: فقط `print` سوال اصلی، پاسخ ادمین و `ticket.get('user_id', 'N/A')` (فیلد `user_id` در تیکت‌ها وجود ندارد). هیچ ایمیل/پیامی واقعاً ارسال نمی‌شود.
3. `add_entries([{"question": ticket["question"], "answer": admin_answer}])`.
4. `tools.close_ticket(question_id)`.

نتیجه یا خطای این تابع به ادمین برنمی‌گردد (فقط لاگ).

### ۸.۱۳ مدیریت پایگاه دانش

**`_get_max_qna_id() -> int`**: همه IDها را با `collection.get(include=[])` می‌گیرد؛ از IDهایی که با `qna-` شروع می‌شوند عدد بعد از خط تیره را parse می‌کند و بیشینه را برمی‌گرداند (اگر نبود ۰). IDهای غیر عددی نادیده گرفته می‌شوند.

**`add_entries(qa_list) -> dict`**:
- لیست خالی ← `{"status": "noop", "message": "No items to add."}`
- زیر `_kb_lock`: `max_id = _get_max_qna_id()`؛ برای آیتم i ام، ID برابر `qna-{max_id + i + 1}`، document/embedding/metadata طبق بخش ۸.۶؛ سپس یک فراخوانی `collection.add(...)` برای همه.
- ← `{"status": "success", "message": "{n} آیتم جدید اضافه شد."}`
- هیچ بررسی تکراری بودن سوال انجام نمی‌شود. اگر بزرگ‌ترین ID حذف شود، ID آن دوباره استفاده می‌شود.

**`_reset_collection() -> int`**: تعداد فعلی را می‌گیرد، `delete_collection` و دوباره `_get_or_create_collection()` (پس کالکشن جدید همیشه **cosine** است)، تعداد قبلی را برمی‌گرداند.

**`overwrite_database(qa_list) -> dict`**: زیر `_kb_lock`: `_reset_collection()` سپس `add_entries(qa_list)` ←
`{"status": "success", "message": "پایگاه دانش بازنویسی شد. تعداد آیتم‌های جدید: {len(qa_list)}"}`. IDها از `qna-1` شروع می‌شوند. عملیات اتمی نیست: اگر امبدینگ وسط کار خطا دهد، پایگاه خالی می‌ماند.

**`delete_entries(ids=None) -> dict`**: زیر `_kb_lock`:
- `ids is None` ← `_reset_collection()` و `{"status": "success", "message": "کل پایگاه دانش ({count} آیتم) پاک شد."}`
- لیست خالی ← `{"status": "noop", "message": "No items to delete."}` (بدون فراخوانی ChromaDB، چون ChromaDB لیست خالی را رد می‌کند)
- وگرنه `collection.delete(ids=ids)` و `{"status": "success", "message": "{len(ids)} آیتم حذف شدند."}` (تعداد درخواستی، نه تعداد واقعاً حذف‌شده؛ IDهای ناموجود خطا نمی‌دهند).

**`get_all_entries() -> list`**: `collection.get(include=["metadatas"])` ← لیست `{"id", "question", "answer"}` (بدون ترتیب تضمین‌شده، بدون صفحه‌بندی).

---

## ۹. سیستم تیکت (`app/services/tools/tools.py`)

- ذخیره در فایل `PENDING_TICKETS_DB = "pending_tickets.json"` (مسیر نسبی به پوشه جاری پروسه).
- قفل سراسری ماژول: `file_lock = threading.Lock()` (فقط داخل یک پروسه محافظت می‌کند).
- فرمت فایل:

```json
{
  "data": [
    {
      "question_id": "uuid",
      "question": "متن سوال",
      "timestamp": "2025-08-11T12:41:47.274143",
      "bot_answer": null,
      "ticket_source": "unanswered"
    }
  ]
}
```

توابع:

| تابع | قفل | رفتار |
|---|---|---|
| `_read_db() -> Dict[str, Dict]` | ندارد (فراخواننده قفل می‌گیرد) | فایل را می‌خواند و به دیکشنری `{question_id: ticket}` تبدیل می‌کند؛ اگر فایل نبود یا JSON خراب بود ← `{}` (بی‌صدا) |
| `_write_db(dict)` | ندارد | کل فایل را با `ensure_ascii=False, indent=2, default=str` بازنویسی می‌کند |
| `list_tickets() -> list` | دارد | همه تیکت‌ها به صورت لیست dict |
| `get_ticket(question_id) -> Optional[Dict]` | دارد | یک تیکت یا `None` |
| `create_ticket(question, bot_answer=None, source="unanswered") -> PendingTicket` | دارد | ساخت `PendingTicket`، ذخیره با `model_dump(mode="json")` (timestamp به ISO string)، برگرداندن خود شیء |
| `close_ticket(question_id) -> bool` | دارد | حذف تیکت؛ `True` اگر وجود داشت، `False` اگر نه |

نکات:
- سوال تکراری تیکت تکراری می‌سازد (بدون dedupe).
- تیکت‌های قدیمی که `bot_answer`/`ticket_source` ندارند یا timestamp آن‌ها به فرمت `"2025-08-11 12:41:47.274143"` (با فاصله) است، هنگام خواندن با پیش‌فرض‌ها و parser پایدانتیک درست خوانده می‌شوند.
- هر عملیات کل فایل را می‌خواند و می‌نویسد (O(n)).

---

## ۱۰. اسکریپت وارد کردن داده (`scripts/import_knowledge_base.py`)

اجرا از ریشه پروژه: `python -m scripts.import_knowledge_base <path.json> [--overwrite]`
1. فایل JSON را می‌خواند. اگر dict بود، کلید `"data"` را برمی‌دارد.
2. فقط آیتم‌هایی که `question` و `answer` غیرخالی دارند را به `{"question", "answer"}` تبدیل می‌کند (بقیه کلیدها مثل `id`، `tags`، `embedding` نادیده گرفته می‌شوند؛ امبدینگ دوباره محاسبه می‌شود).
3. `get_model()` (که کل مدل را بارگذاری می‌کند، پس `.env` لازم است) و سپس `overwrite_database` یا `add_entries`.
4. پیام نتیجه را چاپ می‌کند.

فایل `vector_store.json` داخل `New folder/app.zip` لیستی از ۱۱۵ آبجکت با کلیدهای `id, question, answer, tags, embedding` است و مستقیماً قابل وارد کردن است.

---

## ۱۱. جریان‌های کامل (End-to-End)

**الف) سوال با پاسخ:**
کاربر `POST /api/v1/qna {"query": "..."}` ← جستجوی برداری ← حداقل یک نتیجه با شباهت ≥ آستانه ← LLM پاسخ فارسی ← LLM تگ ← ذخیره در کش ← `200 QAResponse` با `session_id`.

**ب) سوال بدون پاسخ:**
پایگاه خالی یا هیچ نتیجه‌ای بالای آستانه نیست ← تیکت `unanswered` ← `200 TicketCreationResponse` با `question_id`.

**ج) بازخورد منفی:**
`POST /feedback {"session_id", "is_correct": false}` ← تیکت `negative_feedback` با `bot_answer` ← session از کش حذف.

**د) بازخورد مثبت:**
`POST /feedback {"session_id", "is_correct": true}` ← سوال کاربر + پاسخ ربات به‌عنوان آیتم جدید در ChromaDB ← session از کش حذف.

**هـ) چرخه ادمین:**
`GET /admin/pending_tickets` ← ادمین `question_id` را انتخاب می‌کند ← `POST /admin/respond_to_ticket` ← `202` ← در پس‌زمینه: سوال تیکت + پاسخ ادمین به ChromaDB اضافه و تیکت حذف می‌شود ← دفعه بعد که سوال مشابه پرسیده شود، سیستم جواب دارد.

---

## ۱۲. هم‌زمانی و مقیاس‌پذیری

- endpointها در threadpool اجرا می‌شوند؛ قفل‌ها (`_kb_lock`، `_cache_lock`، `file_lock`) فقط **داخل یک پروسه** کار می‌کنند.
- اجرای چند worker (`uvicorn --workers N`) باعث می‌شود: هر worker مدل خودش را بارگذاری کند (حافظه زیاد)، کش بین آن‌ها مشترک نباشد (بازخورد ممکن است «session نامعتبر» بگیرد)، و نوشتن هم‌زمان روی فایل تیکت و ChromaDB محافظت نشود. **سرویس برای یک پروسه طراحی شده است.**
- فراخوانی‌های LLM همگام (blocking) هستند و timeout صریح ندارند (پیش‌فرض کتابخانه openai).
- جستجوی برداری فقط `query` است؛ تولید ID با خواندن همه IDها انجام می‌شود (برای پایگاه‌های بسیار بزرگ کند است).

---

## ۱۳. لاگ‌ها

هیچ `logging` ساختاریافته‌ای وجود ندارد؛ همه چیز با `print` است: پیام‌های `Service:` هنگام بارگذاری، `INFO:` هنگام ساخت/بستن تیکت و افزودن به پایگاه دانش، traceback کامل هنگام خطای OpenAI، و بلوک `SIMULATING:` در پاسخ ادمین. Langfuse پیکربندی شده ولی هیچ tracing واقعی پیاده نشده.

---

## ۱۴. محدودیت‌ها و کارهای باقی‌مانده (برای توسعه بعدی)

1. ارسال واقعی پاسخ ادمین به کاربر پیاده نشده (فقط print)؛ تیکت‌ها `user_id` یا اطلاعات تماس ندارند.
2. بازخورد مثبت بدون بررسی، پاسخ ربات را (حتی پیام خطا/«یافت نشد») به پایگاه دانش اضافه می‌کند.
3. وقتی LLM «پاسخ یافت نشد» می‌گوید، تیکت ساخته نمی‌شود.
4. کش در حافظه است و با ری‌استارت پاک می‌شود؛ پشتیبانی از چند پروسه وجود ندارد.
5. تیکت‌ها در فایل JSON هستند (نه دیتابیس) و dedupe ندارند.
6. پیشوندهای `query:`/`passage:` مدل E5 استفاده نمی‌شوند و امبدینگ آیتم‌ها شامل پاسخ هم هست؛ مقدار مناسب `SIMILARITY_THRESHOLD` باید روی داده واقعی تنظیم شود.
7. Langfuse استفاده نمی‌شود.
8. مسیرهای عمومی rate limit یا احراز هویت ندارند.
9. `get_all_entries` صفحه‌بندی ندارد.
10. تست خودکار (unit/integration) در ریپو وجود ندارد.
11. `@app.on_event("startup")` در FastAPI منسوخ (deprecated) است؛ جایگزین آن `lifespan` است.
12. کلیدهای Langfuse قبلاً در تاریخچه گیت commit شده‌اند و باید باطل شوند.

---

## ۱۵. چک‌لیست بازسازی سریع (برای ایجنت)

1. پروژه FastAPI با ساختار بخش ۳ بساز.
2. `Settings` را دقیقاً طبق جدول بخش ۴ پیاده کن.
3. مدل‌های بخش ۶ را بساز.
4. `QAModel` را به‌صورت singleton ماژولی طبق بخش ۸ پیاده کن (امبدینگ نرمال‌شده، کالکشن cosine، تبدیل فاصله، ID به فرم `qna-n`، کش LRU، قفل‌ها).
5. ماژول تیکت را طبق بخش ۹ پیاده کن.
6. routeها را طبق بخش ۷ با پیشوند `/api/v1` و router ادمین با dependency `X-Admin-Token` وصل کن.
7. متن دقیق promptها و پیام‌های فارسی را از بخش‌های ۸.۷ تا ۸.۱۳ و ۷ کپی کن.
8. اسکریپت import بخش ۱۰ را اضافه کن.
