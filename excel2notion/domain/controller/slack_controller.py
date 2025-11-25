import logging
import requests
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from slack_sdk.errors import SlackApiError

from ..service.excel_service import ExcelService
from ..service.notion_service import NotionService
from ..service.recommendation_service import RecommendationService
from ..repository.slack_repository import SlackRepository
from ..repository.gemini_repository import GeminiRepository
from ..repository.embedding_repository import EmbeddingRepository
from ..config.settings import get_notion_database_id, get_slack_client, get_notion_client
from ..exception.exceptions import ConfigurationException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["slack"])


def get_excel_service() -> ExcelService:
    """ExcelService 인스턴스 생성"""
    slack_client = get_slack_client()
    if not slack_client:
        raise ConfigurationException(detail="Slack 클라이언트가 초기화되지 않았습니다.")
    slack_repo = SlackRepository(slack_client)
    return ExcelService(slack_repo)


def get_notion_service() -> NotionService:
    """NotionService 인스턴스 생성"""
    notion_client = get_notion_client()
    if not notion_client:
        raise ConfigurationException(detail="Notion 클라이언트가 초기화되지 않았습니다.")
    from ..repository.notion_repository import NotionRepository
    notion_repo = NotionRepository(notion_client)
    return NotionService(notion_repo)


def get_recommendation_service() -> RecommendationService:
    """RecommendationService 인스턴스 생성"""
    notion_client = get_notion_client()
    if not notion_client:
        raise ConfigurationException(detail="Notion 클라이언트가 초기화되지 않았습니다.")
    
    gemini_repo = GeminiRepository()
    embedding_repo = EmbeddingRepository(notion_client)
    from ..repository.notion_repository import NotionRepository
    notion_repo = NotionRepository(notion_client)
    
    return RecommendationService(gemini_repo, embedding_repo, notion_repo)


@router.post("/commands")
async def slack_command(request: Request):
    """Slack 슬래시 커맨드 엔드포인트"""
    try:
        form_data = await request.form()
        command = form_data.get("command")
        text = form_data.get("text", "").strip()
        user_id = form_data.get("user_id")
        channel_id = form_data.get("channel_id")
        response_url = form_data.get("response_url")
        
        logger.info(f"Received command: {command}, text: {text}, channel: {channel_id}")
        
        # Slack 클라이언트 가져오기 (모든 커맨드에서 필요)
        slack_client = get_slack_client()
        if not slack_client:
            return JSONResponse(content={
                "response_type": "ephemeral",
                "text": "Slack 클라이언트가 초기화되지 않았습니다. SLACK_BOT_TOKEN을 확인해주세요."
            })
        
        # /append2top1 커맨드 처리
        if command == "/append2top1":
            return await handle_append2top1_command(
                text, channel_id, response_url, slack_client
            )
        
        # /excel2notion 커맨드 처리
        if command != "/excel2notion":
            return JSONResponse(content={
                "response_type": "ephemeral",
                "text": "Unknown command"
            })
        
        # 채널에서 최근 Excel 파일 찾기
        try:
            slack_repo = SlackRepository(slack_client)
            excel_service = ExcelService(slack_repo)
            
            # 채널의 최근 파일 목록 가져오기
            files = slack_repo.list_files(channel_id, file_types="xlsx", count=10)
            
            if not files:
                return JSONResponse(content={
                    "response_type": "ephemeral",
                    "text": "채널에서 Excel 파일(.xlsx)을 찾을 수 없습니다. 파일을 업로드한 후 다시 시도해주세요."
                })
            
            # 가장 최근 파일 사용
            latest_file = files[0]
            file_id = latest_file["id"]
            file_name = latest_file.get("name", "file.xlsx")
            
            # 비동기로 처리 시작 알림
            if response_url:
                requests.post(response_url, json={
                    "response_type": "ephemeral",
                    "text": f"📂 파일 '{file_name}' 처리를 시작합니다..."
                })
            
            # 파일 다운로드 및 처리
            df, _ = excel_service.download_and_parse_file(file_id)
            
            # Notion에 업로드 (속성 자동 생성 활성화)
            db_id = get_notion_database_id()
            if not db_id:
                return JSONResponse(content={
                    "response_type": "ephemeral",
                    "text": "Notion 데이터베이스 ID가 설정되지 않았습니다. NOTION_DATABASE_ID를 확인해주세요."
                })
            
            notion_service = get_notion_service()
            results = notion_service.upload_dataframe(df, db_id, auto_create=True)
            
            # 결과 메시지
            result_text = f"✅ 업로드 완료!\n"
            result_text += f"• 성공: {results['success']}개\n"
            result_text += f"• 실패: {results['failed']}개\n"
            
            if results['auto_created']:
                result_text += f"• 자동 생성된 속성: {', '.join(results['auto_created'])}\n"
            
            if results['unmatched_columns']:
                result_text += f"• 스킵된 컬럼: {', '.join(results['unmatched_columns'])}\n"
            
            if results['errors'] and len(results['errors']) <= 5:
                result_text += f"\n오류:\n" + "\n".join(results['errors'])
            elif results['errors']:
                result_text += f"\n오류: {len(results['errors'])}개 발생 (처음 3개: {', '.join(results['errors'][:3])})"
            
            return JSONResponse(content={
                "response_type": "in_channel",
                "text": result_text
            })
            
        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            return JSONResponse(content={
                "response_type": "ephemeral",
                "text": f"❌ Slack API 오류: {e.response['error']}"
            })
        
    except Exception as e:
        logger.error(f"Error processing command: {str(e)}")
        return JSONResponse(content={
            "response_type": "ephemeral",
            "text": f"❌ 오류가 발생했습니다: {str(e)}"
        })


@router.post("/events")
async def slack_events(request: Request):
    """Slack 이벤트 엔드포인트 (파일 업로드 처리)"""
    try:
        data = await request.json()
        
        # URL verification challenge
        if data.get("type") == "url_verification":
            return JSONResponse(content={"challenge": data.get("challenge")})
        
        # 이벤트 처리
        event = data.get("event", {})
        event_type = event.get("type")
        
        # 메시지에 파일이 첨부된 경우 처리
        if event_type == "message" and "files" in event:
            files = event.get("files", [])
            for file_info in files:
                file_name = file_info.get("name", "")
                if file_name.endswith((".xlsx", ".xls")):
                    file_id = file_info.get("id")
                    channel_id = event.get("channel")
                    
                    logger.info(f"Excel file detected: {file_name}, file_id: {file_id}")
                    
                    # 비동기로 처리
                    try:
                        slack_client = get_slack_client()
                        notion_client = get_notion_client()
                        db_id = get_notion_database_id()
                        
                        if slack_client and notion_client and db_id:
                            slack_repo = SlackRepository(slack_client)
                            excel_service = ExcelService(slack_repo)
                            
                            df, _ = excel_service.download_and_parse_file(file_id)
                            
                            from ..repository.notion_repository import NotionRepository
                            notion_repo = NotionRepository(notion_client)
                            notion_service = NotionService(notion_repo)
                            results = notion_service.upload_dataframe(df, db_id, auto_create=True)
                            
                            # 결과를 채널에 메시지로 전송
                            result_text = f"✅ Excel 파일 '{file_name}'이 Notion에 업로드되었습니다!\n"
                            result_text += f"성공: {results['success']}개, 실패: {results['failed']}개"
                            
                            if results['auto_created']:
                                result_text += f"\n자동 생성된 속성: {', '.join(results['auto_created'])}"
                            
                            slack_repo.post_message(channel_id, result_text)
                    except Exception as e:
                        logger.error(f"Error processing file: {str(e)}")
                        if slack_client:
                            slack_repo = SlackRepository(slack_client)
                            slack_repo.post_message(
                                channel_id,
                                f"❌ 파일 처리 중 오류가 발생했습니다: {str(e)}"
                            )
        
        return JSONResponse(content={"status": "ok"})
        
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        return JSONResponse(content={"status": "error", "message": str(e)})


async def handle_append2top1_command(
    text: str,
    channel_id: str,
    response_url: Optional[str],
    slack_client
):
    """append2top1 커맨드 처리"""
    try:
        slack_repo = SlackRepository(slack_client)
        recommendation_service = get_recommendation_service()
        
        # 텍스트 입력이 있으면 사용, 없으면 파일 찾기
        file_content = None
        if not text:
            # 채널의 최근 파일 목록 가져오기 (PDF)
            files = slack_repo.list_files(channel_id, file_types="pdf", count=10)
            
            if not files:
                return JSONResponse(content={
                    "response_type": "ephemeral",
                    "text": "텍스트를 입력하거나 PDF 파일을 업로드해주세요."
                })
            
            # 가장 최근 파일 사용
            latest_file = files[0]
            file_id = latest_file["id"]
            file_name = latest_file.get("name", "file.pdf")
            
            # 비동기로 처리 시작 알림
            if response_url:
                requests.post(response_url, json={
                    "response_type": "ephemeral",
                    "text": f"📂 파일 '{file_name}' 처리를 시작합니다..."
                })
            
            # 파일 다운로드
            file_content, _ = slack_repo.download_file(file_id)
        else:
            # 비동기로 처리 시작 알림
            if response_url:
                requests.post(response_url, json={
                    "response_type": "ephemeral",
                    "text": "📝 텍스트를 분석하고 있습니다..."
                })
        
        # Notion 데이터베이스 ID 가져오기
        db_id = get_notion_database_id()
        if not db_id:
            return JSONResponse(content={
                "response_type": "ephemeral",
                "text": "Notion 데이터베이스 ID가 설정되지 않았습니다. NOTION_DATABASE_ID를 확인해주세요."
            })
        
        # 추천 프로세스 실행
        result = recommendation_service.process_append2top1(
            text=text if text else None,
            file_content=file_content,
            database_id=db_id
        )
        
        # 결과 메시지 생성
        result_text = f"✅ 추천 완료!\n\n"
        result_text += f"🏆 추천 식당: {result['top1_restaurant']}\n"
        result_text += f"🍺 추천 주류: {result.get('recommended_drink', '소주')}\n"
        result_text += f"📊 유사도 점수: {result['similarity_score']:.4f}\n\n"
        result_text += f"💡 추천 근거:\n{result['recommendation_reason']}\n\n"
        
        # Top1 식당의 추천 이유와 유사도
        if 'top1_recommendation_reason' in result and 'reason_similarity' in result:
            result_text += f"📝 Top1 식당 추천 이유:\n{result['top1_recommendation_reason']}\n\n"
            result_text += f"🔗 추천 이유 유사도: {result['reason_similarity']:.4f}\n\n"
        
        result_text += f"➕ 새로 추가된 행 ID: {result['new_page_id']}\n"
        
        # Slack에 알림 전송
        slack_repo.post_message(channel_id, result_text)
        
        return JSONResponse(content={
            "response_type": "in_channel",
            "text": result_text
        })
        
    except Exception as e:
        logger.error(f"Error processing append2top1 command: {str(e)}")
        return JSONResponse(content={
            "response_type": "ephemeral",
            "text": f"❌ 오류가 발생했습니다: {str(e)}"
        })

