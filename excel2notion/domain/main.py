import os
import tempfile
from typing import Optional, Tuple
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from notion_client import Client
from dotenv import load_dotenv
import requests
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


# 환경변수 로드
load_dotenv()


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Excel to Notion Slack Bot")


# 환경변수에서 설정 가져오기
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")


# Notion 클라이언트 초기화
notion = None
if NOTION_TOKEN:
    notion = Client(auth=NOTION_TOKEN)


# Slack 클라이언트 초기화
slack_client = None
if SLACK_BOT_TOKEN:
    slack_client = WebClient(token=SLACK_BOT_TOKEN)



def download_file_from_slack(file_id: str) -> Tuple[bytes, str]:
    """Slack 파일 ID로 파일 다운로드"""
    if not slack_client:
        raise HTTPException(status_code=500, detail="Slack client not initialized")
    
    try:
        # 파일 정보 가져오기
        file_info = slack_client.files_info(file=file_id)
        file_data = file_info["file"]
        file_name = file_data.get("name", "file.xlsx")
        file_url_private = file_data.get("url_private_download")
        
        if not file_url_private:
            raise HTTPException(status_code=400, detail="File download URL not available")
        
        # requests로 직접 다운로드
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        response = requests.get(file_url_private, headers=headers)
        response.raise_for_status()
        file_content = response.content
        
        return file_content, file_name
    except SlackApiError as e:
        logger.error(f"Slack API error: {e.response['error']}")
        raise HTTPException(status_code=500, detail=f"Slack API error: {e.response['error']}")



def download_file_from_url(file_url: str, token: str) -> bytes:
    """Slack 파일 URL에서 파일 다운로드 (대체 방법)"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(file_url, headers=headers)
    response.raise_for_status()
    return response.content



def parse_excel_file(file_content: bytes) -> pd.DataFrame:
    """Excel 파일을 pandas DataFrame으로 변환"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name
    
    try:
        df = pd.read_excel(tmp_file_path, engine='openpyxl')
        return df
    finally:
        os.unlink(tmp_file_path)



def get_notion_database_properties(database_id: str) -> dict:
    """Notion Database의 속성 정보 가져오기"""
    if not notion:
        raise HTTPException(status_code=500, detail="Notion client not initialized")
    
    try:
        db_info = notion.databases.retrieve(database_id=database_id)
        properties = db_info.get("properties", {})
        
        # 속성명 → 속성타입 매핑
        property_map = {}
        for prop_name, prop_info in properties.items():
            property_map[prop_name] = prop_info.get("type")
        
        return property_map
    except Exception as e:
        logger.error(f"Failed to retrieve database properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Notion DB 조회 실패: {str(e)}")



def smart_detect_property_type(column_name: str, sample_values: list) -> dict:
    """컬럼명과 샘플 데이터로 속성 타입 자동 추론"""
    column_lower = column_name.lower()
    
    # 컬럼명 기반 타입 추론
    if any(keyword in column_lower for keyword in ["가격", "금액", "price", "cost", "amount"]):
        return {"number": {"format": "won"}}
    
    if any(keyword in column_lower for keyword in ["평점", "점수", "rating", "score"]):
        return {"number": {"format": "number"}}
    
    if any(keyword in column_lower for keyword in ["날짜", "date", "day"]):
        return {"date": {}}
    
    if any(keyword in column_lower for keyword in ["url", "링크", "link"]):
        return {"url": {}}
    
    if any(keyword in column_lower for keyword in ["이메일", "email", "mail"]):
        return {"email": {}}
    
    if any(keyword in column_lower for keyword in ["전화", "phone", "tel"]):
        return {"phone_number": {}}
    
    # 샘플 데이터 기반 타입 추론
    non_null_values = [v for v in sample_values if pd.notna(v)]
    
    if not non_null_values:
        return {"rich_text": {}}
    
    # 모두 숫자인지 확인
    try:
        all_numbers = all(isinstance(v, (int, float)) or str(v).replace('.', '', 1).replace('-', '', 1).isdigit() 
                         for v in non_null_values[:10])
        if all_numbers:
            return {"number": {}}
    except:
        pass
    
    # 모두 True/False인지 확인
    try:
        all_bools = all(str(v).lower() in ['true', 'false', '0', '1', 'yes', 'no'] 
                       for v in non_null_values[:10])
        if all_bools:
            return {"checkbox": {}}
    except:
        pass
    
    # 기본값: rich_text
    return {"rich_text": {}}



def auto_create_notion_properties(database_id: str, df: pd.DataFrame) -> dict:
    """엑셀 컬럼 분석 후 Notion DB 속성 자동 생성 (스마트 타입 추론)"""
    if not notion:
        raise HTTPException(status_code=500, detail="Notion client not initialized")
    
    try:
        # 기존 DB 정보 가져오기
        db_info = notion.databases.retrieve(database_id=database_id)
        existing_properties = db_info.get("properties", {})
        
        # Title 속성 존재 여부 확인
        has_title = any(prop.get("type") == "title" for prop in existing_properties.values())
        
        # 새로 추가할 속성 정의
        new_properties = {}
        
        for idx, col in enumerate(df.columns):
            if col in existing_properties:
                logger.info(f"Property '{col}' already exists, skipping...")
                continue
            
            # 샘플 데이터 추출 (처음 20개)
            sample_values = df[col].head(20).tolist()
            
            # 첫 번째 컬럼이고 Title이 없으면 Title로 설정
            if idx == 0 and not has_title:
                new_properties[col] = {"title": {}}
                has_title = True
                logger.info(f"Setting '{col}' as Title property")
            else:
                # 스마트 타입 추론
                prop_type = smart_detect_property_type(col, sample_values)
                new_properties[col] = prop_type
                logger.info(f"Detected type for '{col}': {list(prop_type.keys())[0]}")
        
        if not new_properties:
            logger.info("No new properties to add")
            return {"status": "success", "added": [], "types": {}}
        
        # Notion DB 업데이트
        logger.info(f"Adding {len(new_properties)} properties: {list(new_properties.keys())}")
        notion.databases.update(
            database_id=database_id,
            properties=new_properties
        )
        
        return {
            "status": "success",
            "added": list(new_properties.keys()),
            "types": {k: list(v.keys())[0] for k, v in new_properties.items()}
        }
    
    except Exception as e:
        logger.error(f"Failed to auto-create properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"속성 자동 생성 실패: {str(e)}")



def convert_to_notion_properties(row: pd.Series, df: pd.DataFrame, db_properties: dict) -> dict:
    """DataFrame의 행을 Notion 속성 형식으로 변환 (자동 매핑)"""
    properties = {}
    
    for col in df.columns:
        value = row[col]
        
        # NaN 값 스킵
        if pd.isna(value):
            continue
        
        # Notion DB에 해당 컬럼명이 존재하는지 확인
        if col not in db_properties:
            logger.warning(f"Column '{col}' not found in Notion DB properties, skipping...")
            continue
        
        prop_type = db_properties[col]
        
        # 속성 타입별로 변환
        if prop_type == "title":
            properties[col] = {
                "title": [
                    {
                        "text": {
                            "content": str(value)[:2000]  # Notion title 길이 제한
                        }
                    }
                ]
            }
        elif prop_type == "rich_text":
            properties[col] = {
                "rich_text": [
                    {
                        "text": {
                            "content": str(value)[:2000]  # Notion text 길이 제한
                        }
                    }
                ]
            }
        elif prop_type == "number":
            try:
                properties[col] = {
                    "number": float(value)
                }
            except (ValueError, TypeError):
                logger.warning(f"Cannot convert '{value}' to number for column '{col}'")
                continue
        elif prop_type == "checkbox":
            properties[col] = {
                "checkbox": bool(value)
            }
        elif prop_type == "date":
            try:
                if pd.notna(value):
                    date_str = pd.Timestamp(value).strftime("%Y-%m-%d")
                    properties[col] = {
                        "date": {
                            "start": date_str
                        }
                    }
            except Exception as e:
                logger.warning(f"Cannot convert '{value}' to date for column '{col}': {e}")
                continue
        elif prop_type == "select":
            properties[col] = {
                "select": {
                    "name": str(value)[:100]  # Select option 길이 제한
                }
            }
        elif prop_type == "multi_select":
            # 쉼표로 구분된 값을 multi_select로 변환
            options = [opt.strip() for opt in str(value).split(",") if opt.strip()]
            properties[col] = {
                "multi_select": [
                    {"name": opt[:100]} for opt in options[:10]  # 최대 10개 옵션
                ]
            }
        elif prop_type == "url":
            properties[col] = {
                "url": str(value)[:2000]
            }
        elif prop_type == "email":
            properties[col] = {
                "email": str(value)[:200]
            }
        elif prop_type == "phone_number":
            properties[col] = {
                "phone_number": str(value)[:50]
            }
        else:
            # 지원하지 않는 타입은 rich_text로 변환
            logger.warning(f"Unsupported property type '{prop_type}' for column '{col}', converting to rich_text")
            properties[col] = {
                "rich_text": [
                    {
                        "text": {
                            "content": str(value)[:2000]
                        }
                    }
                ]
            }
    
    return properties



def upload_to_notion(df: pd.DataFrame, database_id: str, auto_create: bool = True) -> dict:
    """DataFrame 데이터를 Notion 데이터베이스에 업로드 (속성 자동 생성 옵션)"""
    if not notion:
        raise HTTPException(status_code=500, detail="Notion client not initialized")
    
    # 속성 자동 생성
    creation_result = {"added": [], "types": {}}
    if auto_create:
        logger.info("Auto-creating Notion properties from Excel columns...")
        creation_result = auto_create_notion_properties(database_id, df)
        logger.info(f"Property creation result: {creation_result}")
    
    # Notion DB 속성 정보 가져오기
    logger.info("Fetching Notion database properties...")
    db_properties = get_notion_database_properties(database_id)
    logger.info(f"Found {len(db_properties)} properties: {list(db_properties.keys())}")
    
    # 엑셀 컬럼 중 매핑 가능한 것들 확인
    excel_cols = set(df.columns)
    notion_props = set(db_properties.keys())
    matched_cols = excel_cols & notion_props
    unmatched_cols = excel_cols - notion_props
    
    logger.info(f"Matched columns: {matched_cols}")
    if unmatched_cols:
        logger.warning(f"Unmatched columns (will be skipped): {unmatched_cols}")
    
    results = {
        "success": 0,
        "failed": 0,
        "errors": [],
        "matched_columns": list(matched_cols),
        "unmatched_columns": list(unmatched_cols),
        "auto_created": creation_result.get("added", []),
        "auto_created_types": creation_result.get("types", {})
    }
    
    for idx, row in df.iterrows():
        try:
            properties = convert_to_notion_properties(row, df, db_properties)
            
            if not properties:
                results["failed"] += 1
                error_msg = f"Row {idx + 1} failed: No valid properties to upload"
                results["errors"].append(error_msg)
                logger.warning(error_msg)
                continue
            
            # Notion에 페이지 생성
            notion.pages.create(
                parent={"database_id": database_id},
                properties=properties
            )
            results["success"] += 1
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Uploaded {idx + 1} rows...")
                
        except Exception as e:
            results["failed"] += 1
            error_msg = f"Row {idx + 1} failed: {str(e)}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
    
    return results



@app.post("/slack/commands")
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
        
        if command != "/excel2notion":
            return JSONResponse(content={
                "response_type": "ephemeral",
                "text": "Unknown command"
            })
        
        # 채널에서 최근 Excel 파일 찾기
        if not slack_client:
            return JSONResponse(content={
                "response_type": "ephemeral",
                "text": "Slack 클라이언트가 초기화되지 않았습니다. SLACK_BOT_TOKEN을 확인해주세요."
            })
        
        try:
            # 채널의 최근 파일 목록 가져오기
            files_response = slack_client.files_list(
                channel=channel_id,
                types="xlsx",
                count=10
            )
            
            files = files_response.get("files", [])
            
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
            file_content, _ = download_file_from_slack(file_id)
            
            # Excel 파일 파싱
            df = parse_excel_file(file_content)
            logger.info(f"Excel file parsed: {len(df)} rows, {len(df.columns)} columns")
            
            # Notion에 업로드 (속성 자동 생성 활성화)
            db_id = NOTION_DATABASE_ID
            if not db_id:
                return JSONResponse(content={
                    "response_type": "ephemeral",
                    "text": "Notion 데이터베이스 ID가 설정되지 않았습니다. NOTION_DATABASE_ID를 확인해주세요."
                })
            
            results = upload_to_notion(df, db_id, auto_create=True)
            
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



@app.post("/slack/events")
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
                        file_content, _ = download_file_from_slack(file_id)
                        df = parse_excel_file(file_content)
                        
                        db_id = NOTION_DATABASE_ID
                        if db_id and notion:
                            results = upload_to_notion(df, db_id, auto_create=True)
                            
                            # 결과를 채널에 메시지로 전송
                            if slack_client:
                                result_text = f"✅ Excel 파일 '{file_name}'이 Notion에 업로드되었습니다!\n"
                                result_text += f"성공: {results['success']}개, 실패: {results['failed']}개"
                                
                                if results['auto_created']:
                                    result_text += f"\n자동 생성된 속성: {', '.join(results['auto_created'])}"
                                
                                slack_client.chat_postMessage(
                                    channel=channel_id,
                                    text=result_text
                                )
                    except Exception as e:
                        logger.error(f"Error processing file: {str(e)}")
                        if slack_client:
                            slack_client.chat_postMessage(
                                channel=channel_id,
                                text=f"❌ 파일 처리 중 오류가 발생했습니다: {str(e)}"
                            )
        
        return JSONResponse(content={"status": "ok"})
        
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        return JSONResponse(content={"status": "error", "message": str(e)})



@app.post("/upload-excel")
async def upload_excel(
    file_url: Optional[str] = Form(None),
    slack_token: Optional[str] = Form(None),
    database_id: Optional[str] = Form(None)
):
    """Excel 파일을 업로드하고 Notion에 전송"""
    try:
        if not file_url:
            raise HTTPException(status_code=400, detail="file_url is required")
        
        if not slack_token:
            raise HTTPException(status_code=400, detail="slack_token is required")
        
        db_id = database_id or NOTION_DATABASE_ID
        if not db_id:
            raise HTTPException(status_code=400, detail="database_id is required")
        
        # 파일 다운로드
        logger.info(f"Downloading file from: {file_url}")
        file_content = download_file_from_url(file_url, slack_token)
        
        # Excel 파일 파싱
        logger.info("Parsing Excel file...")
        df = parse_excel_file(file_content)
        logger.info(f"Excel file parsed: {len(df)} rows, {len(df.columns)} columns")
        
        # Notion에 업로드 (속성 자동 생성)
        logger.info("Uploading to Notion...")
        results = upload_to_notion(df, db_id, auto_create=True)
        
        message = f"업로드 완료: 성공 {results['success']}개, 실패 {results['failed']}개"
        if results['auto_created']:
            message += f" | 자동 생성된 속성: {', '.join(results['auto_created'])}"
        
        return JSONResponse(content={
            "status": "success",
            "message": message,
            "details": results
        })
        
    except Exception as e:
        logger.error(f"Error uploading Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/health")
async def health_check():
    """Health check 엔드포인트"""
    return JSONResponse(content={
        "status": "healthy",
        "notion_configured": notion is not None,
        "slack_configured": slack_client is not None
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

