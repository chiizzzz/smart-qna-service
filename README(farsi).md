# 🧠 سیستم پرسش و پاسخ هوشمند (RAG)

این پروژه یک سرویس وب با **FastAPI** است که با معماری RAG به سوالات کاربران بر اساس یک پایگاه دانش پاسخ می‌دهد. سوال با یک مدل امبدینگ چندزبانه به بردار تبدیل می‌شود، در پایگاه دانش برداری **ChromaDB** جستجو می‌شود و یک مدل زبانی (از طریق هر API سازگار با **OpenAI**) پاسخ فارسی تولید می‌کند. سوالاتی که پاسخ ندارند به تیکت تبدیل می‌شوند و پاسخ اپراتور دوباره به پایگاه دانش اضافه می‌شود.

## ✨ ویژگی‌ها

- **هسته RAG:** جستجوی معنایی (شباهت کسینوسی) در ChromaDB و تولید پاسخ فقط بر اساس اطلاعات بازیابی‌شده.
- **تگ‌گذاری خودکار:** هر سوال در یکی از دسته‌های پشتیبانی طبقه‌بندی می‌شود.
- **تیکتینگ:** سوالات بی‌پاسخ و پاسخ‌هایی که کاربر اشتباه اعلام کرده به تیکت تبدیل می‌شوند.
- **حلقه بازخورد:** بازخورد مثبت و پاسخ اپراتور به پایگاه دانش اضافه می‌شود.
- **API ادمین امن:** مسیرهای مدیریت پایگاه دانش و تیکت‌ها به توکن ادمین نیاز دارند.
- **مستندات خودکار** در آدرس `/docs`.

## 🚀 راه‌اندازی

### ۱. نصب

```bash
git clone https://github.com/chiizzzz/smart-qna-service.git
cd smart-qna-service
python -m venv .venv
source .venv/bin/activate        # ویندوز: .\.venv\Scripts\Activate
pip install -r requirements.txt
```

### ۲. تنظیمات

```bash
cp .env.example .env
```

حداقل `OPENAI_API_KEY` و `ADMIN_API_KEY` را مقداردهی کنید. اگر `ADMIN_API_KEY` خالی باشد، همه مسیرهای ادمین خطای `503` برمی‌گردانند.

### ۳. اجرا

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### ۴. پر کردن پایگاه دانش

پوشه `./chroma_db_store` در اولین اجرا ساخته می‌شود و در گیت ذخیره نمی‌شود. پایگاه دانش را از طریق API ادمین یا از یک فایل JSON (لیستی از `{"question": ..., "answer": ...}`) پر کنید:

```bash
python -m scripts.import_knowledge_base path/to/data.json            # افزودن
python -m scripts.import_knowledge_base path/to/data.json --overwrite # جایگزینی کامل
```

فایل `vector_store.json` قدیمی داخل `New folder/app.zip` هم قابل استفاده است.

## ⚙️ API

### مسیرهای عمومی

| متد | مسیر | توضیح |
|---|---|---|
| `POST` | `/api/v1/qna` | پرسیدن سوال: `{"query": "..."}` |
| `POST` | `/api/v1/feedback` | ثبت بازخورد: `{"session_id": "...", "is_correct": true}` |

### مسیرهای ادمین (هدر `X-Admin-Token: <ADMIN_API_KEY>`)

| متد | مسیر | توضیح |
|---|---|---|
| `POST` | `/api/v1/admin/knowledge_base/add` | افزودن دسته‌ای: `{"data": [{"question": "...", "answer": "..."}]}` |
| `POST` | `/api/v1/admin/knowledge_base/add_from_operator` | افزودن یک پرسش و پاسخ |
| `PUT` | `/api/v1/admin/knowledge_base/overwrite` | بازنویسی کامل پایگاه دانش |
| `DELETE` | `/api/v1/admin/knowledge_base/delete` | حذف با `{"ids": [...]}`، یا حذف همه اگر ids ارسال نشود |
| `GET` | `/api/v1/admin/knowledge_base/all` | لیست همه آیتم‌ها |
| `GET` | `/api/v1/admin/pending_tickets` | لیست تیکت‌های در انتظار |
| `POST` | `/api/v1/admin/respond_to_ticket` | پاسخ به تیکت: `{"question_id": "...", "answer": "..."}` |

**مثال:**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "هزینه سفر چگونه محاسبه می‌شود؟"}' \
  http://127.0.0.1:8000/api/v1/qna
```

## 🔧 نکات تنظیمات

- `SIMILARITY_THRESHOLD` شباهت کسینوسی است (پیش‌فرض `0.75`). کالکشن‌هایی که قبلاً ساخته شده‌اند فضای L2 دارند؛ سرویس این را تشخیص می‌دهد و فاصله را درست تبدیل می‌کند. برای تبدیل به کسینوسی، داده‌ها را با `--overwrite` دوباره وارد کنید.
- `LLM_MAX_TOKENS` (پیش‌فرض `512`) حداکثر طول پاسخ را تعیین می‌کند.
