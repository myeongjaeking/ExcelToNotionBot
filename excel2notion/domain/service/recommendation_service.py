import logging
from typing import Dict, Tuple, Optional, List
from ..repository.gemini_repository import GeminiRepository
from ..repository.embedding_repository import EmbeddingRepository
from ..repository.notion_repository import NotionRepository
from ..utils.embedding_utils import (
    create_weighted_embedding,
    get_column_weights,
    create_embedding,
    cosine_similarity
)
from ..utils.pdf_parser import extract_text_from_input
from ..utils.property_converter import convert_to_notion_properties
from ..config.faiss_manager import get_faiss_manager
import pandas as pd

logger = logging.getLogger(__name__)


class RecommendationService:
    """추천 관련 비즈니스 로직을 담당하는 Service"""
    
    def __init__(
        self,
        gemini_repo: GeminiRepository,
        embedding_repo: EmbeddingRepository,
        notion_repo: NotionRepository
    ):
        self.gemini_repo = gemini_repo
        self.embedding_repo = embedding_repo
        self.notion_repo = notion_repo
    
    def process_append2top1(
        self,
        text: Optional[str],
        file_content: Optional[bytes],
        database_id: str
    ) -> Dict:
        """append2top1 전체 프로세스 실행"""
        try:
            # 1. 입력값 분석 (자연어/PDF 판별 및 텍스트 추출)
            logger.info("Extracting text from input...")
            extracted_text = extract_text_from_input(file_content, text)
            
            # 2. Gemini로 정해진 클래스로 변환
            logger.info("Classifying text with Gemini...")
            structured_data = self.gemini_repo.classify_to_structured_data(extracted_text)
            logger.info(f"Structured data: {structured_data}")
            
            # 3. 열 가중치 적용하여 임베딩 생성
            logger.info("Creating weighted embedding...")
            weights = get_column_weights()
            query_embedding = create_weighted_embedding(structured_data, weights)
            
            # 4. FAISS 인덱스에서 검색
            logger.info("Searching in FAISS index...")
            faiss_manager = get_faiss_manager()
            
            # FAISS 인덱스가 비어있으면 Notion에서 데이터를 가져와서 인덱스 생성
            if faiss_manager.index is None or faiss_manager.index.ntotal == 0:
                logger.warning("FAISS index is empty. Loading from Notion and building index...")
                self._build_faiss_index_from_notion(database_id)
            
            # FAISS에서 검색
            results = faiss_manager.search(query_embedding, k=1)
            
            if not results:
                raise ValueError("검색 결과가 없습니다.")
            
            top1_page_id, top1_data, similarity_score = results[0]
            
            # 6. 추천 주류 및 추천 이유 생성
            # Top1 식당의 기존 추천 주류를 그대로 사용
            logger.info("Getting recommended drink from top1 restaurant...")
            recommended_drink = top1_data.get("추천주류", top1_data.get("추천 주류", ""))
            
            # 추천 주류가 없으면 기본값 사용
            if not recommended_drink:
                logger.warning("No recommended drink found in top1 data, using default")
                recommended_drink = "소주"
            
            # Top1 식당의 기존 추천 이유 가져오기 (Notion에서 또는 생성)
            logger.info("Getting top1 restaurant existing recommendation reason...")
            top1_recommendation_reason = top1_data.get("추천 이유", top1_data.get("추천이유", ""))
            
            # 기존 추천 이유가 없으면 생성
            if not top1_recommendation_reason:
                logger.info("No existing recommendation reason found, generating new one...")
                top1_recommendation_reason = self.gemini_repo.generate_top1_recommendation_reason(top1_data)
            
            # 새로운 추천 이유 생성 (소믈리에 스타일)
            logger.info("Generating recommendation reason with sommelier style...")
            # structured_data에 추천주류 추가
            structured_data_with_drink = structured_data.copy()
            structured_data_with_drink["추천주류"] = recommended_drink
            
            # top1_data에 추천주류가 없으면 추가
            top1_data_with_drink = top1_data.copy()
            if "추천주류" not in top1_data_with_drink and "추천 주류" not in top1_data_with_drink:
                top1_drink = top1_data.get("추천주류", top1_data.get("추천 주류", recommended_drink))
                top1_data_with_drink["추천주류"] = top1_drink
            
            recommendation_reason = self.gemini_repo.generate_recommendation_reason(
                test_row=structured_data_with_drink,
                similar_train_row=top1_data_with_drink,
                existing_reason=top1_recommendation_reason,
                similarity_score=similarity_score
            )
            
            # 두 추천 이유 간의 유사도 계산
            logger.info("Calculating similarity between recommendation reasons...")
            reason_embedding1 = create_embedding(recommendation_reason)
            reason_embedding2 = create_embedding(top1_recommendation_reason)
            reason_similarity = cosine_similarity(reason_embedding1, reason_embedding2)
            
            # 7. 결과를 Notion에 저장
            logger.info("Saving result to Notion...")
            db_properties = self.notion_repo.get_database_properties(database_id)
            
            # structured_data의 키를 Notion DB 필드명에 맞게 변환
            # "음식 종류" -> "음식종류" 또는 그 반대
            structured_data_normalized = structured_data.copy()
            if "음식 종류" in structured_data_normalized:
                # Notion DB에 "음식종류"가 있으면 키 변경
                if "음식종류" in db_properties and "음식 종류" not in db_properties:
                    structured_data_normalized["음식종류"] = structured_data_normalized.pop("음식 종류")
                # Notion DB에 "음식 종류"가 있으면 그대로 유지
            elif "음식종류" in structured_data_normalized:
                # Notion DB에 "음식 종류"가 있으면 키 변경
                if "음식 종류" in db_properties and "음식종류" not in db_properties:
                    structured_data_normalized["음식 종류"] = structured_data_normalized.pop("음식종류")
            
            # DataFrame 생성 (단일 행)
            df = pd.DataFrame([structured_data_normalized])
            properties = convert_to_notion_properties(df.iloc[0], df, db_properties)
            
            # 음식종류가 properties에 없으면 명시적으로 추가 (필드명 변형 처리)
            food_type = structured_data.get("음식 종류", structured_data.get("음식종류", ""))
            if food_type and "음식 종류" not in properties and "음식종류" not in properties:
                if "음식 종류" in db_properties:
                    properties["음식 종류"] = {
                        "rich_text": [
                            {
                                "text": {
                                    "content": str(food_type)
                                }
                            }
                        ]
                    }
                elif "음식종류" in db_properties:
                    properties["음식종류"] = {
                        "rich_text": [
                            {
                                "text": {
                                    "content": str(food_type)
                                }
                            }
                        ]
                    }
            
            # 추천 주류와 추천 이유를 properties에 추가
            if "추천 주류" in db_properties:
                properties["추천 주류"] = {
                    "rich_text": [
                        {
                            "text": {
                                "content": recommended_drink
                            }
                        }
                    ]
                }
            elif "추천주류" in db_properties:
                properties["추천주류"] = {
                    "rich_text": [
                        {
                            "text": {
                                "content": recommended_drink
                            }
                        }
                    ]
                }
            
            if "추천 이유" in db_properties:
                properties["추천 이유"] = {
                    "rich_text": [
                        {
                            "text": {
                                "content": recommendation_reason[:2000]  # Notion 길이 제한
                            }
                        }
                    ]
                }
            elif "추천이유" in db_properties:
                properties["추천이유"] = {
                    "rich_text": [
                        {
                            "text": {
                                "content": recommendation_reason[:2000]
                            }
                        }
                    ]
                }
            
            # 임베딩과 함께 페이지 생성
            new_page_id = self.embedding_repo.create_page_with_embedding(
                database_id,
                properties,
                query_embedding
            )
            
            # FAISS 인덱스에 새 페이지 추가
            logger.info("Adding new page to FAISS index...")
            faiss_manager = get_faiss_manager()
            faiss_manager.add_embedding(new_page_id, query_embedding, structured_data)
            faiss_manager.save_index()
            logger.info("New page added to FAISS index and saved")
            
            return {
                "success": True,
                "similarity_score": similarity_score,
                "top1_restaurant": top1_data.get("식당명", "알 수 없음"),
                "recommended_drink": recommended_drink,
                "recommendation_reason": recommendation_reason,
                "top1_recommendation_reason": top1_recommendation_reason,
                "reason_similarity": reason_similarity,
                "new_page_id": new_page_id,
                "structured_data": structured_data,
                "top1_data": top1_data
            }
            
        except Exception as e:
            logger.error(f"Error in process_append2top1: {str(e)}")
            raise
    
    def _build_faiss_index_from_notion(self, database_id: str):
        """Notion에서 모든 페이지를 가져와서 FAISS 인덱스 구축"""
        try:
            logger.info("Fetching all pages from Notion database...")
            # 모든 페이지 가져오기 (임베딩 포함 시도)
            pages_with_embeddings = self.embedding_repo.get_all_pages_with_embeddings(database_id)
            
            # 임베딩이 없는 페이지가 있으면 생성
            all_pages = self.embedding_repo.get_all_pages(database_id)
            existing_page_ids = {page_id for page_id, _, _ in pages_with_embeddings}
            
            weights = get_column_weights()
            faiss_manager = get_faiss_manager()
            
            # 임베딩이 없는 페이지에 대해 임베딩 생성
            for page_id, page_data in all_pages:
                if page_id in existing_page_ids:
                    continue  # 이미 임베딩이 있는 페이지는 스킵
                
                try:
                    # 페이지 데이터에서 필요한 컬럼 추출
                    data_for_embedding = {}
                    for column in weights.keys():
                        if column in page_data and weights[column] > 0:
                            data_for_embedding[column] = page_data[column]
                    
                    if not data_for_embedding:
                        logger.warning(f"Page {page_id} has no data for embedding, skipping...")
                        continue
                    
                    # 가중치 적용하여 임베딩 생성
                    embedding = create_weighted_embedding(data_for_embedding, weights)
                    
                    # FAISS 인덱스에만 추가 (Notion에는 저장하지 않음 - 길이 제한 때문)
                    faiss_manager.add_embedding(page_id, embedding, page_data)
                    
                    pages_with_embeddings.append((page_id, page_data, embedding))
                    logger.info(f"Created embedding and added to FAISS for page {page_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to create embedding for page {page_id}: {str(e)}")
                    continue
            
            # 이미 임베딩이 있는 페이지들을 FAISS에 추가
            for page_id, page_data, embedding in pages_with_embeddings:
                if page_id not in faiss_manager.get_all_page_ids():
                    faiss_manager.add_embedding(page_id, embedding, page_data)
            
            # FAISS 인덱스 저장
            faiss_manager.save_index()
            logger.info(f"Successfully built FAISS index with {len(pages_with_embeddings)} pages")
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index from Notion: {str(e)}")
            raise ValueError(f"FAISS 인덱스 구축 실패: {str(e)}")

