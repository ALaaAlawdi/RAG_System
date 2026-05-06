import os
from app.core.config import settings  
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from app.api.v1.router import api_router

for _dir in [settings.UPLOADS_DIR, settings.OCR_INPUTS_DIR, settings.OCR_OUTPUTS_DIR, "chroma_db"]:
    os.makedirs(_dir, exist_ok=True)

app = FastAPI(title="RAG System", version="1.0.0", docs_url=None, redoc_url=None)
app.include_router(api_router, prefix="/api/v1")


@app.get("/docs", include_in_schema=False)
async def swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="RAG System - Swagger UI",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_ui():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="RAG System - ReDoc",
        redoc_js_url="https://unpkg.com/redoc@latest/bundles/redoc.standalone.js",
    )
