import logging
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# SentenceTransformer 모델 초기화 (전역 변수로 한 번만 로드)
_model = None


def get_embedding_model():
    """SentenceTransformer 모델 가져오기 (싱글톤 패턴)"""
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model: dragonkue/multilingual-e5-small-ko-v2")
        _model = SentenceTransformer('dragonkue/multilingual-e5-small-ko-v2')
    return _model


# 열 가중치 정의
COLUMN_WEIGHTS = {
    "시그니처메뉴": 0.3333,
    "음식 종류": 0.2778,
    "평균가격": 0.2778,
    "지역": 0.0556,
    "평점": 0.0556,
    "식당명": 0.0
}


def get_column_weights() -> Dict[str, float]:
    """열 가중치 반환"""
    return COLUMN_WEIGHTS.copy()


def create_embedding(text: str) -> List[float]:
    """SentenceTransformer를 사용하여 텍스트 임베딩 생성"""
    try:
        if not text or not text.strip():
            raise ValueError("텍스트가 비어있습니다.")
        
        model = get_embedding_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Failed to create embedding: {str(e)}")
        raise ValueError(f"임베딩 생성 실패: {str(e)}")


def create_weighted_embedding(data: Dict[str, any], weights: Dict[str, float]) -> List[float]:
    """가중치를 적용한 임베딩 생성"""
    embeddings = {}
    
    # 각 열에 대해 임베딩 생성
    for column, value in data.items():
        if column in weights and weights[column] > 0:
            text = str(value) if value is not None else ""
            if text:
                embeddings[column] = create_embedding(text)
    
    # 가중치 적용하여 결합
    if not embeddings:
        raise ValueError("임베딩을 생성할 수 있는 데이터가 없습니다.")
    
    # 임베딩 차원 확인
    dim = len(list(embeddings.values())[0])
    weighted_embedding = np.zeros(dim)
    
    for column, embedding in embeddings.items():
        weight = weights.get(column, 0.0)
        weighted_embedding += np.array(embedding) * weight
    
    # 정규화
    norm = np.linalg.norm(weighted_embedding)
    if norm > 0:
        weighted_embedding = weighted_embedding / norm
    
    return weighted_embedding.tolist()


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """코사인 유사도 계산"""
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def find_top1_similar(
    query_embedding: List[float],
    database_embeddings: List[Tuple[str, List[float]]]
) -> Tuple[str, float]:
    """가장 유사한 항목 찾기 (Top 1)"""
    if not database_embeddings:
        raise ValueError("데이터베이스 임베딩이 없습니다.")
    
    max_similarity = -1
    top_id = None
    
    for page_id, embedding in database_embeddings:
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            top_id = page_id
    
    return top_id, max_similarity

