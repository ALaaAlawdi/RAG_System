from fastapi import APIRouter, Depends, HTTPException
from ....knowledge_center.center import KnowledgeCenter, UPLOADS_DIR
from ....schemas.ask import AskRequest, AskResponse
from ....core.logger import setup_logger
from ...dependencies import get_knowledge_center

logger = setup_logger(__name__)
router = APIRouter()


@router.post("/", response_model=AskResponse)
async def ask_question(
    body: AskRequest,
    kc: KnowledgeCenter = Depends(get_knowledge_center),
):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if body.file_path and not body.file_path.startswith(UPLOADS_DIR):
        raise HTTPException(
            status_code=400, detail=f"file_path must start with '{UPLOADS_DIR}/'."
        )

    result = await kc.chat(
        query=body.query,
        k=body.k,
        file_path=body.file_path,
        provider=body.provider,
    )
    
    if result is None:
        raise HTTPException(status_code=500, detail="An error occurred while processing your query.")

    return AskResponse(answer=result["answer"], sources=result["sources"])
