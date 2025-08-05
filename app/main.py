from fastapi import FastAPI
from app.core.config import settings
from app.api import routes

app = FastAPI(
    title="Smart Q&A Service",
    description="A RAG-based question and answering service using Llama-3.",
    version="0.1.0",
)

# چاپ مقدار prefix برای اطمینان
print("✅ API_PREFIX =", settings.API_PREFIX)

# mount کردن مسیر اصلی API
app.include_router(routes.router, prefix=settings.API_PREFIX)


# مسیر روت ساده برای تست سریع
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Welcome to the Smart Q&A API. Go to /docs for interactive documentation."
    }


# مسیر تستی برای بررسی ارتباط با سرور
@app.get("/ping")
def ping():
    print("📡 [MAIN] پینگ از route اصلی")
    return {"status": "pong"}


# هنگام شروع برنامه، همه route ها رو چاپ کن
@app.on_event("startup")
async def show_all_routes():
    print("\n📍 [ROUTES] مسیرهای فعال FastAPI:")
    for route in app.router.routes:
        print(f"🔗 {route.path} → {route.name}")
