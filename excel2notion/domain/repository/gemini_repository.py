import logging
import json
from anthropic import Anthropic
from typing import Dict, Any
from ..config.settings import get_claude_api_key
from ..exception.exceptions import ConfigurationException

logger = logging.getLogger(__name__)

class ClaudeRepository:
    """Claude Sonnet 4 API 접근을 담당하는 Repository"""
    
    def __init__(self):
        api_key = get_claude_api_key()
        if not api_key:
            raise ConfigurationException(detail="Claude API 키가 설정되지 않았습니다.")
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def classify_to_structured_data(self, text: str) -> Dict[str, Any]:
        """자연어/PDF 텍스트를 구조화된 데이터로 변환"""
        prompt = f"""다음 텍스트를 분석하여 식당 정보를 추출하세요. 다음 형식의 JSON으로 응답하세요:
{{
    "시그니처메뉴": "시그니처 메뉴 이름",
    "음식 종류": "음식 종류 (예: 한식, 중식, 일식, 양식 등)",
    "평균가격": 숫자 (예: 15000),
    "지역": "지역명",
    "평점": 숫자 (0-5 사이),
    "식당명": "식당 이름"
}}

텍스트:
{text}

JSON 형식으로만 응답하세요. 정보가 없으면 null을 사용하세요."""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            result_text = message.content[0].text.strip()
            
            # JSON 추출 (마크다운 코드 블록 제거)
            if "```json" in result_text:
                result_text = result_text.split("```json").split("```").strip()[1]
            elif "```" in result_text:
                result_text = result_text.split("```")[3].split("```")[0].strip()
            
            return json.loads(result_text)
        except Exception as e:
            logger.error(f"Failed to classify text with Claude: {str(e)}")
            raise ValueError(f"Claude 분류 실패: {str(e)}")
    
    def generate_recommendation_reason(
        self, 
        test_row: Dict[str, Any], 
        similar_train_row: Dict[str, Any], 
        existing_reason: str,
        similarity_score: float
    ) -> str:
        """추천 이유 생성 (소믈리에 스타일)"""
        # 유사도 점수를 퍼센트로 변환 (0-100)
        similarity_percent = int(similarity_score * 100)
        
        prompt = f"""당신은 소믈리에입니다.

[필수 규칙 - 절대 변경 금지]

다음 철학/뉘앙스를 모든 문장에 반드시 포함하세요:

{existing_reason}

추천 근거를 {similarity_percent}% 반영하고, {100-similarity_percent}%는 새로운 내용을 작성하세요.

[신규 요청]

식당명: {test_row.get('식당명', '')}
지역: {test_row.get('지역', '')}
음식종류: {test_row.get('음식 종류', test_row.get('음식종류', ''))}
평균가격: {test_row.get('평균가격', test_row.get('평균 가격', ''))}
평점: {test_row.get('평점', '')}
시그니처메뉴: {test_row.get('시그니처메뉴', test_row.get('시그니처 메뉴', ''))}
추천주류: {test_row.get('추천주류', test_row.get('추천 주류', ''))}

[매칭된 B2B 데이터 ({similarity_percent}% 반영)]

식당명: {similar_train_row.get('식당명', '')}
지역: {similar_train_row.get('지역', '')}
음식종류: {similar_train_row.get('음식 종류', similar_train_row.get('음식종류', ''))}
평균가격: {similar_train_row.get('평균가격', similar_train_row.get('평균 가격', ''))}
평점: {similar_train_row.get('평점', '')}
시그니처메뉴: {similar_train_row.get('시그니처메뉴', similar_train_row.get('시그니처 메뉴', ''))}
추천주류: {similar_train_row.get('추천주류', similar_train_row.get('추천 주류', ''))}

B2B 이유: {existing_reason}

규칙:
- {similarity_percent}%: B2B 이유의 뉘앙스 유지
- {100-similarity_percent}%: 새로운 분석 추가
- 50글자 이내"""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            reason = message.content[0].text.strip()
            # 50글자 제한 적용
            if len(reason) > 50:
                reason = reason[:50]
            return reason
        except Exception as e:
            logger.error(f"Failed to generate recommendation reason: {str(e)}")
            return f"유사도 {similarity_percent}%로 추천되었습니다."
    
    def extract_recommended_drink(self, top1_data: Dict[str, Any]) -> str:
        """추천 식당의 주류 추출"""
        prompt = f"""다음 식당 정보를 바탕으로 이 식당에 어울리는 주류를 추천해주세요.

식당 정보:
{top1_data}

음식 종류, 시그니처 메뉴, 지역 등을 고려하여 가장 어울리는 주류 하나만 추천해주세요.
예: 소주, 맥주, 와인, 막걸리, 전통주 등

주류 이름만 간단히 답변해주세요 (설명 없이)."""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            drink = message.content[0].text.strip()
            # 설명이 포함된 경우 첫 번째 단어만 추출
            drink = drink.split()[0] if drink else "소주"
            return drink
        except Exception as e:
            logger.error(f"Failed to extract recommended drink: {str(e)}")
            return "소주"  # 기본값
    
    def generate_top1_recommendation_reason(self, top1_data: Dict[str, Any]) -> str:
        """Top1 식당의 추천 이유 생성 (기존 추천 이유)"""
        prompt = f"""다음 식당 정보를 바탕으로 이 식당을 추천하는 이유를 작성해주세요.

식당 정보:
{top1_data}

이 식당의 특징, 메뉴, 평점 등을 바탕으로 왜 이 식당을 추천하는지 간결하게 설명해주세요.
한국어로 2-3문장으로 작성해주세요."""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Failed to generate top1 recommendation reason: {str(e)}")
            return "맛있는 음식과 좋은 서비스를 제공하는 식당입니다."
