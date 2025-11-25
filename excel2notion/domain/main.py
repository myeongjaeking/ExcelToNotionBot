from fastapi import FastAPI
from contextlib import asynccontextmanager
from .controller import slack_router, upload_router, health_router
from .config.faiss_manager import initialize_faiss_index
from .config.settings import get_notion_database_id
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행"""
    # 서버 시작 시
    logger.info("Starting up server...")
    try:
        database_id = get_notion_database_id()
        initialize_faiss_index(database_id)
        logger.info("Server startup completed")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
    
    yield
    
    # 서버 종료 시 (필요시 정리 작업)
    logger.info("Shutting down server...")


app = FastAPI(
    title="Excel to Notion Slack Bot",
    lifespan=lifespan
)

# 라우터 등록
app.include_router(slack_router)
app.include_router(upload_router)
app.include_router(health_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
