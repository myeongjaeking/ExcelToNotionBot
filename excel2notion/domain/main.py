from fastapi import FastAPI
from .controller import slack_router, upload_router, health_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Excel to Notion Slack Bot")

# 라우터 등록
app.include_router(slack_router)
app.include_router(upload_router)
app.include_router(health_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
