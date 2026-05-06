from fastapi import APIRouter
from .endpoints import upload, ask

api_router = APIRouter()

api_router.include_router(upload.router, prefix="/upload", tags=["Upload"])
api_router.include_router(ask.router, prefix="/ask", tags=["Ask"])
