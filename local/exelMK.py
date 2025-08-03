# --- START OF FILE exelMK.py ---

import pandas as pd
import re

DATA_FILE = 'faq_data.txt'
OUTPUT_FILE = 'input.xlsx'


def parse_qa_from_text(text):
    """
    متن خام را به لیستی از دیکشنری‌های پرسش و پاسخ تبدیل می‌کند.
    این نسخه کمی قوی‌تر و خواناتر است.
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    qa_pairs = []

    ignore_list = [
        "بخش مسافران", "پرداخت و عضویت", "بخش رانندگان", "نحوه ثبت سفارش",
        "رویه‌های بازگرداندن کالا", "شیوه‌های پرداخت", "سفارش", "کد تخفیف",
        "محدوده", "پرداخت", "معاملات", "کیف پول داخلی", "فروش به سایت",
        "حساب کاربری", "تایید دو مرحله ای", "امنیت و پشتیبانی", "احراز هویت",
        "حساب کاربری ورود و ثبت نام"
    ]

    i = 0
    while i < len(lines):
        question = lines[i]
        is_heading = any(title == question for title in ignore_list)

        # اگر خط فعلی عنوان بخش است، آن را رد کن
        if is_heading:
            i += 1
            continue

        # یک خط را به عنوان پرسش در نظر می‌گیریم اگر با "سوال:" شروع شود یا با ؟ تمام شود.
        if question.startswith('سوال:'):
            question = question.replace('سوال:', '').strip()
        elif not question.endswith('؟'):
            i += 1
            continue  # اگر نه سوال است و نه عنوان، رد شو

        # حالا شروع به جمع‌آوری پاسخ کن
        answer_parts = []
        i += 1
        while i < len(lines):
            next_line = lines[i]
            # اگر به یک سوال یا عنوان جدید رسیدیم، جمع‌آوری پاسخ تمام است.
            is_next_line_question = next_line.endswith('؟') or next_line.startswith('سوال:') or any(
                title == next_line for title in ignore_list)

            if is_next_line_question:
                break

            # بخش پاسخ را اضافه کن
            cleaned_part = next_line.replace('پاسخ:', '').strip()
            answer_parts.append(cleaned_part)
            i += 1

        full_answer = "\n".join(answer_parts)

        if question and full_answer:
            qa_pairs.append({'سوال': question, 'پاسخ': full_answer})

    return qa_pairs


# --- اجرای اصلی ---
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        text_data = f.read()
except FileNotFoundError:
    print(f"خطا: فایل داده '{DATA_FILE}' یافت نشد. لطفاً آن را ایجاد کنید.")
    exit()

qa_data = parse_qa_from_text(text_data)
df = pd.DataFrame(qa_data)

try:
    df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    print(f"فایل با {len(df)} پرسش و پاسخ با موفقیت در '{OUTPUT_FILE}' ذخیره شد.")
except Exception as e:
    print(f"خطا در ذخیره‌سازی فایل اکسل: {e}")

# --- END OF FILE exelMK.py ---