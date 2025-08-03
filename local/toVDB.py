# --- START OF FILE toVDB.py ---

import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

EXCEL_FILE = "output_tagged_llama.xlsx"
VECTOR_DB_FILE = "vector_store.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

print(f"در حال بارگذاری مدل امبدینگ: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

try:
    df = pd.read_excel(EXCEL_FILE)
except FileNotFoundError:
    print(f"خطا: فایل ورودی '{EXCEL_FILE}' یافت نشد. ابتدا اسکریپت TagVDB.py را اجرا کنید.")
    exit()

required_columns = ["سوال", "پاسخ", "تگ‌های Llama3"]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"ستون‌های موردنیاز در فایل اکسل یافت نشدند: {required_columns}")

vector_db = []

# استفاده از iterrows برای دسترسی بهینه به سطرها
for idx, row in tqdm(df.iterrows(), total=len(df), desc="در حال ساخت پایگاه دانش برداری"):
    question = str(row["سوال"]).strip()
    answer = str(row["پاسخ"]).strip()
    tags_raw = str(row.get("تگ‌های Llama3", "")).strip()  # استفاده از .get برای جلوگیری از خطا

    # (تغییر مهم) ترکیب پرسش و پاسخ برای امبدینگ بهتر
    text_to_embed = f"سوال: {question}\n\nپاسخ: {answer}"
    embedding = model.encode(text_to_embed, convert_to_numpy=True).tolist()

    # تمیز کردن تگ‌ها
    tags = [tag.strip() for tag in tags_raw.split(',') if tag.strip()]

    vector_db.append({
        "id": f"qna-{idx}",
        "question": question,
        "answer": answer,
        "tags": tags,
        "embedding": embedding
    })

with open(VECTOR_DB_FILE, "w", encoding="utf-8") as f:
    json.dump(vector_db, f, ensure_ascii=False, indent=2)

print(f"✅ {len(vector_db)} رکورد با موفقیت در '{VECTOR_DB_FILE}' ذخیره شد.")

# --- END OF FILE toVDB.py ---