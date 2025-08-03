from fastapi import FastAPI
from app.core.config import settings
from app.api import routes
app = FastAPI(
    title="Smart Q&A Service",
    description="A RAG-based question and answering service using Llama-3.",
    version="0.1.0",
)
app.include_router(routes.router, prefix=settings.API_PREFIX)
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Smart Q&A API. Go to /docs for interactive documentation."}