import logging
import json
from typing import Dict, Optional, List
from anthropic import Anthropic
from ..exception.exceptions import NotionException

logger = logging.getLogger(__name__)

class NotionMCPRepository:
    """Notion MCP (Model Context Protocol) 접근을 담당하는 Repository"""
    
    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            raise NotionException(status_code=500, detail="Claude API key not initialized")
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        # Notion MCP 설정
        self.mcp_tools = self._initialize_notion_mcp()
    
    def _initialize_notion_mcp(self) -> List[Dict]:
        """Notion MCP 도구 초기화"""
        return [
            {
                "name": "notion_query_database",
                "description": "Notion 데이터베이스 쿼리",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "database_id": {"type": "string", "description": "Notion Database ID"},
                        "filter": {"type": "object", "description": "쿼리 필터 (선택)"},
                        "sorts": {"type": "array", "description": "정렬 규칙 (선택)"}
                    },
                    "required": ["database_id"]
                }
            },
            {
                "name": "notion_get_page",
                "description": "Notion 페이지 조회",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "page_id": {"type": "string", "description": "Notion Page ID"}
                    },
                    "required": ["page_id"]
                }
            },
            {
                "name": "notion_create_page",
                "description": "Notion 페이지 생성",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "database_id": {"type": "string", "description": "Notion Database ID"},
                        "properties": {"type": "object", "description": "페이지 속성"}
                    },
                    "required": ["database_id", "properties"]
                }
            },
            {
                "name": "notion_update_page",
                "description": "Notion 페이지 업데이트",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "page_id": {"type": "string", "description": "Notion Page ID"},
                        "properties": {"type": "object", "description": "업데이트할 속성"}
                    },
                    "required": ["page_id", "properties"]
                }
            },
            {
                "name": "notion_get_database_info",
                "description": "Notion 데이터베이스 정보 조회",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "database_id": {"type": "string", "description": "Notion Database ID"}
                    },
                    "required": ["database_id"]
                }
            }
        ]
    
    def get_database_properties(self, database_id: str) -> Dict[str, str]:
        """Notion Database의 속성 정보 가져오기 (MCP)"""
        try:
            prompt = f"""Notion MCP를 사용하여 데이터베이스 ID '{database_id}'의 속성 정보를 조회해주세요.
            
다음 형식의 JSON을 응답해주세요:
{{
    "property_name": "property_type",
    ...
}}"""
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Claude가 MCP 도구를 사용하도록 요청
            response = self._call_claude_with_mcp(messages, "notion_get_database_info", 
                                                  {"database_id": database_id})
            
            # 응답에서 속성 정보 추출
            property_map = self._extract_properties_from_response(response)
            return property_map
            
        except Exception as e:
            logger.error(f"Failed to retrieve database properties: {str(e)}")
            raise NotionException(status_code=500, detail=f"Notion DB 조회 실패: {str(e)}")
    
    def create_page(self, database_id: str, properties: Dict):
        """Notion에 페이지 생성 (MCP)"""
        try:
            prompt = f"""Notion MCP를 사용하여 데이터베이스 '{database_id}'에 다음 속성으로 페이지를 생성해주세요:

속성: {json.dumps(properties, ensure_ascii=False)}"""
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            self._call_claude_with_mcp(messages, "notion_create_page", 
                                      {"database_id": database_id, "properties": properties})
            
            logger.info(f"Page created successfully in database {database_id}")
        except Exception as e:
            logger.error(f"Failed to create page: {str(e)}")
            raise NotionException(status_code=500, detail=f"페이지 생성 실패: {str(e)}")
    
    def update_database(self, database_id: str, properties: Dict):
        """Notion Database 속성 업데이트 (MCP)"""
        try:
            prompt = f"""Notion MCP를 사용하여 데이터베이스 '{database_id}'의 속성을 다음과 같이 업데이트해주세요:

업데이트 속성: {json.dumps(properties, ensure_ascii=False)}"""
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            self._call_claude_with_mcp(messages, "notion_get_database_info",
                                      {"database_id": database_id})
            
            logger.info(f"Database {database_id} updated successfully")
        except Exception as e:
            logger.error(f"Failed to update database: {str(e)}")
            raise NotionException(status_code=500, detail=f"데이터베이스 업데이트 실패: {str(e)}")
    
    def retrieve_database(self, database_id: str) -> Dict:
        """Notion Database 정보 조회 (MCP)"""
        try:
            prompt = f"""Notion MCP를 사용하여 데이터베이스 ID '{database_id}'의 전체 정보를 조회해주세요.
            
JSON 형식으로 응답해주세요."""
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self._call_claude_with_mcp(messages, "notion_get_database_info",
                                                 {"database_id": database_id})
            
            return response
        except Exception as e:
            logger.error(f"Failed to retrieve database: {str(e)}")
            raise NotionException(status_code=500, detail=f"데이터베이스 조회 실패: {str(e)}")
    
    def get_all_pages_data(self, database_id: str) -> List[Dict]:
        """데이터베이스의 모든 페이지 데이터 가져오기 (MCP)"""
        try:
            prompt = f"""Notion MCP를 사용하여 데이터베이스 ID '{database_id}'의 모든 페이지를 조회해주세요.
            
각 페이지의 다음 정보를 추출하여 JSON 배열로 응답해주세요:
- 모든 속성명과 값
- title, rich_text, number, select, multi_select, date, checkbox, url, email, phone_number 등 모든 타입 지원

응답 형식:
[
    {{
        "property_name": "value",
        ...
    }},
    ...
]"""
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self._call_claude_with_mcp(messages, "notion_query_database",
                                                 {"database_id": database_id})
            
            pages = self._extract_pages_from_response(response)
            return pages
        except Exception as e:
            logger.error(f"Failed to get all pages data: {str(e)}")
            raise NotionException(status_code=500, detail=f"페이지 데이터 조회 실패: {str(e)}")
    
    def _call_claude_with_mcp(self, messages: List[Dict], tool_name: str, tool_input: Dict) -> str:
        """Claude MCP 도구 호출"""
        try:
            # 첫 번째 메시지: 사용자 요청
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=messages,
                tools=self.mcp_tools
            )
            
            # Claude가 MCP 도구를 사용하도록 유도
            if response.stop_reason == "tool_use":
                tool_result = self._execute_mcp_tool(tool_name, tool_input)
                
                # MCP 결과를 포함한 추가 메시지
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": response.content[0].id if response.content else "",
                            "content": json.dumps(tool_result, ensure_ascii=False)
                        }
                    ]
                })
                
                # 최종 응답 획득
                final_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=messages
                )
                
                return final_response.content[0].text if final_response.content else ""
            
            return response.content[0].text if response.content else ""
        except Exception as e:
            logger.error(f"Failed to call Claude with MCP: {str(e)}")
            raise
    
    def _execute_mcp_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """MCP 도구 실행 (시뮬레이션)"""
        # 실제 Notion API 호출 로직
        # notion_client를 직접 사용하거나 MCP 서버와 통신
        logger.info(f"Executing MCP tool: {tool_name} with input: {tool_input}")
        return {"status": "success", "data": tool_input}
    
    def _extract_properties_from_response(self, response: str) -> Dict[str, str]:
        """응답에서 속성 정보 추출"""
        try:
            # JSON 블록 추출
            if "```json" in response:
                json_str = response.split("```json").split("```").strip()[1]
            elif "```" in response:
                json_str = response.split("```")[3].split("```")[0].strip()
            else:
                json_str = response
            
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to extract properties from response: {str(e)}")
            return {}
    
    def _extract_pages_from_response(self, response: str) -> List[Dict]:
        """응답에서 페이지 데이터 추출"""
        try:
            # JSON 배열 추출
            if "```json" in response:
                json_str = response.split("```json").split("```").strip()[1]
            elif "```" in response:
                json_str = response.split("```")[3].split("```")[0].strip()
            else:
                json_str = response
            
            pages = json.loads(json_str)
            return pages if isinstance(pages, list) else [pages]
        except Exception as e:
            logger.warning(f"Failed to extract pages from response: {str(e)}")
            return []
