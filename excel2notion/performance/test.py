import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# 1. 모델 로드
print("🔄 모델 로드 중...")
model = SentenceTransformer('dragonkue/multilingual-e5-small-ko-v2')
print("모델 로드 완료\n")


# 2. 데이터 로드
df = pd.read_excel("Restaurant_Recommendation_Final.xlsx")
df.columns = [col.strip().lower() for col in df.columns]
print(f" 총 {len(df)}개 행 로드")

train_data = df.iloc[0:800].reset_index(drop=True)
test_data = df.iloc[800:1000].reset_index(drop=True)

print(f"\n📊 Train: {len(train_data)}개 | Test: {len(test_data)}개")

def get_text_baseline(row):
    """기본: 모든 특성 포함"""
    parts = []
    
    if '음식종류' in row.index and pd.notna(row.get('음식종류')):
        parts.append(str(row.get('음식종류', '')))
    
    if '시그니처메뉴' in row.index and pd.notna(row.get('시그니처메뉴')):
        menu = str(row.get('시그니처메뉴', ''))[:50]
        parts.append(menu)
    
    if '평균가격' in row.index and pd.notna(row.get('평균가격')):
        try:
            price = str(int(row.get('평균가격', '')))
            parts.append(price)
        except:
            pass
    
    if '추천이유' in row.index and pd.notna(row.get('추천이유')):
        reason = str(row.get('추천이유', ''))[:100]
        parts.append(reason)
    
    text = " ".join(parts)
    return text if text else "정보 없음"


def get_text_with_restaurant(row):
    """식당명 포함"""
    parts = []
    
    # 식당명 추가
    if '식당명' in row.index and pd.notna(row.get('식당명')):
        parts.append(str(row.get('식당명', '')).strip())
    
    if '음식종류' in row.index and pd.notna(row.get('음식종류')):
        parts.append(str(row.get('음식종류', '')))
    
    if '시그니처메뉴' in row.index and pd.notna(row.get('시그니처메뉴')):
        menu = str(row.get('시그니처메뉴', ''))[:50]
        parts.append(menu)
    
    if '평균가격' in row.index and pd.notna(row.get('평균가격')):
        try:
            price = str(int(row.get('평균가격', '')))
            parts.append(price)
        except:
            pass
    
    if '추천이유' in row.index and pd.notna(row.get('추천이유')):
        reason = str(row.get('추천이유', ''))[:100]
        parts.append(reason)
    
    text = " ".join(parts)
    return text if text else "정보 없음"


def get_text_with_region(row):
    """지역 포함"""
    parts = []
    
    # 지역 추가
    if '지역' in row.index and pd.notna(row.get('지역')):
        parts.append(str(row.get('지역', '')).strip())
    
    if '음식종류' in row.index and pd.notna(row.get('음식종류')):
        parts.append(str(row.get('음식종류', '')))
    
    if '시그니처메뉴' in row.index and pd.notna(row.get('시그니처메뉴')):
        menu = str(row.get('시그니처메뉴', ''))[:50]
        parts.append(menu)
    
    if '평균가격' in row.index and pd.notna(row.get('평균가격')):
        try:
            price = str(int(row.get('평균가격', '')))
            parts.append(price)
        except:
            pass
    
    if '추천이유' in row.index and pd.notna(row.get('추천이유')):
        reason = str(row.get('추천이유', ''))[:100]
        parts.append(reason)
    
    text = " ".join(parts)
    return text if text else "정보 없음"


def get_text_with_both(row):
    """식당명 + 지역 포함"""
    parts = []
    
    # 식당명 추가
    if '식당명' in row.index and pd.notna(row.get('식당명')):
        parts.append(str(row.get('식당명', '')).strip())
    
    # 지역 추가
    if '지역' in row.index and pd.notna(row.get('지역')):
        parts.append(str(row.get('지역', '')).strip())
    
    if '음식종류' in row.index and pd.notna(row.get('음식종류')):
        parts.append(str(row.get('음식종류', '')))
    
    if '시그니처메뉴' in row.index and pd.notna(row.get('시그니처메뉴')):
        menu = str(row.get('시그니처메뉴', ''))[:50]
        parts.append(menu)
    
    if '평균가격' in row.index and pd.notna(row.get('평균가격')):
        try:
            price = str(int(row.get('평균가격', '')))
            parts.append(price)
        except:
            pass
    
    if '추천이유' in row.index and pd.notna(row.get('추천이유')):
        reason = str(row.get('추천이유', ''))[:100]
        parts.append(reason)
    
    text = " ".join(parts)
    return text if text else "정보 없음"


def get_text_price_menu_only(row):
    """평균가격 + 시그니처메뉴만"""
    parts = []
    
    if '시그니처메뉴' in row.index and pd.notna(row.get('시그니처메뉴')):
        menu = str(row.get('시그니처메뉴', ''))[:50]
        parts.append(menu)
    
    if '평균가격' in row.index and pd.notna(row.get('평균가격')):
        try:
            price = str(int(row.get('평균가격', '')))
            parts.append(price)
        except:
            pass
    
    text = " ".join(parts)
    return text if text else "정보 없음"


def get_text_foodtype_menu_only(row):
    """음식종류 + 시그니처메뉴만"""
    parts = []
    
    if '음식종류' in row.index and pd.notna(row.get('음식종류')):
        parts.append(str(row.get('음식종류', '')))
    
    if '시그니처메뉴' in row.index and pd.notna(row.get('시그니처메뉴')):
        menu = str(row.get('시그니처메뉴', ''))[:50]
        parts.append(menu)
    
    text = " ".join(parts)
    return text if text else "정보 없음"


def get_text_price_foodtype_only(row):
    """평균가격 + 음식종류만"""
    parts = []
    
    if '음식종류' in row.index and pd.notna(row.get('음식종류')):
        parts.append(str(row.get('음식종류', '')))
    
    if '평균가격' in row.index and pd.notna(row.get('평균가격')):
        try:
            price = str(int(row.get('평균가격', '')))
            parts.append(price)
        except:
            pass
    
    text = " ".join(parts)
    return text if text else "정보 없음"


def get_text_with_rating(row):
    """평점 포함"""
    parts = []
    
    # 평점 추가
    if '평점' in row.index and pd.notna(row.get('평점')):
        parts.append(str(row.get('평점', '')).strip())
    
    if '음식종류' in row.index and pd.notna(row.get('음식종류')):
        parts.append(str(row.get('음식종류', '')))
    
    if '시그니처메뉴' in row.index and pd.notna(row.get('시그니처메뉴')):
        menu = str(row.get('시그니처메뉴', ''))[:50]
        parts.append(menu)
    
    if '평균가격' in row.index and pd.notna(row.get('평균가격')):
        try:
            price = str(int(row.get('평균가격', '')))
            parts.append(price)
        except:
            pass
    
    if '추천이유' in row.index and pd.notna(row.get('추천이유')):
        reason = str(row.get('추천이유', ''))[:100]
        parts.append(reason)
    
    text = " ".join(parts)
    return text if text else "정보 없음"


# 4. 실험 설정
experiments = [
    {
        'name': 'Baseline (음식종류+시그니처메뉴+평균가격)',
        'get_text_func': get_text_baseline,
        'cache_file': 'train_embeddings_baseline.pkl'
    },
    {
        'name': '평균가격+시그니처메뉴',
        'get_text_func': get_text_price_menu_only,
        'cache_file': 'train_embeddings_price_menu_only.pkl'
    },
    {
        'name': '음식종류+시그니처메뉴',
        'get_text_func': get_text_foodtype_menu_only,
        'cache_file': 'train_embeddings_foodtype_menu_only.pkl'
    },
    {
        'name': '평균가격+음식종류',
        'get_text_func': get_text_price_foodtype_only,
        'cache_file': 'train_embeddings_price_foodtype_only.pkl'
    },
    {
        'name': '식당명 포함',
        'get_text_func': get_text_with_restaurant,
        'cache_file': 'train_embeddings_with_restaurant.pkl'
    },
    {
        'name': '지역 포함',
        'get_text_func': get_text_with_region,
        'cache_file': 'train_embeddings_with_region.pkl'
    },
    {
        'name': '평점 포함',
        'get_text_func': get_text_with_rating,
        'cache_file': 'train_embeddings_with_rating.pkl'
    }
]


# 5. 각 실험 실행
results = []

for exp_idx, exp in enumerate(experiments):
    print("\n" + "="*100)
    print(f"🔬 실험 {exp_idx + 1}/{len(experiments)}: {exp['name']}")
    print("="*100)
    
    # Train 임베딩 생성 (캐싱)
    try:
        print(f"\n💾 캐시에서 Train 임베딩 로드...")
        with open(exp['cache_file'], 'rb') as f:
            cache = pickle.load(f)
            train_embeddings = cache['embeddings']
            train_wines = cache['wines']
        print("✅ 캐시 로드 완료")
    except:
        print(f"\n🔄 Train 임베딩 생성 중...")
        train_texts = [exp['get_text_func'](row) for _, row in train_data.iterrows()]
        train_embeddings = model.encode(train_texts, show_progress_bar=True)
        
        train_wines = []
        for _, row in train_data.iterrows():
            wine = None
            if '추천주류' in row.index and pd.notna(row.get('추천주류')):
                wine_str = str(row.get('추천주류', '')).strip()
                wine_list = [w.strip() for w in wine_str.split(',')]
                wine = wine_list[0] if wine_list else None
            train_wines.append(wine)
        
        with open(exp['cache_file'], 'wb') as f:
            pickle.dump({'embeddings': train_embeddings, 'wines': train_wines}, f)
        print("✅ Train 임베딩 생성 및 캐시 저장")
    
    # Test 데이터 추천
    print(f"\n🔮 Test 데이터 추천 중...")
    top_1_correct = 0
    
    for idx, test_row in test_data.iterrows():
        test_text = exp['get_text_func'](test_row)
        test_embedding = model.encode([test_text], show_progress_bar=False)[0]
        
        similarities = cosine_similarity(
            test_embedding.reshape(1, -1),
            train_embeddings
        )[0]
        
        top_5_idx = np.argsort(similarities)[-5:][::-1]
        top_5_similarities = similarities[top_5_idx]
        
        wine_scores = {}
        weights = [0.35, 0.25, 0.20, 0.15, 0.05]
        
        for i, train_idx in enumerate(top_5_idx):
            wine = train_wines[train_idx]
            if wine:
                score = weights[i] * top_5_similarities[i]
                if wine not in wine_scores or score > wine_scores[wine]:
                    wine_scores[wine] = score
        
        if wine_scores:
            recommended_wines = sorted(wine_scores.items(), 
                                       key=lambda x: x[1], reverse=True)[:3]
            recommended_wines = [w[0] for w in recommended_wines]
        else:
            recommended_wines = []
        
        actual_wine = None
        if '추천주류' in test_row.index and pd.notna(test_row.get('추천주류')):
            wine_str = str(test_row.get('추천주류', '')).strip()
            wine_list = [w.strip() for w in wine_str.split(',')]
            actual_wine = wine_list[0] if wine_list else None
        
        if actual_wine and recommended_wines and recommended_wines[0] == actual_wine:
            top_1_correct += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(test_data)} 완료")
    
    # 정확도 계산
    total = len(test_data)
    accuracy = (top_1_correct / total * 100) if total > 0 else 0
    
    results.append({
        '실험명': exp['name'],
        '정확도': accuracy,
        '정확개수': top_1_correct,
        '전체개수': total
    })
    
    print(f"\n📊 {exp['name']} 정확도: {accuracy:.2f}% ({top_1_correct}/{total})")


# 6. 기여도 분석
print("\n" + "="*100)
print("📊 Ablation Study 결과 및 기여도 분석")
print("="*100)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# 기여도 계산
baseline_acc = results[0]['정확도']  # Baseline (음식종류+시그니처메뉴+평균가격)
price_menu_acc = results[1]['정확도']  # 평균가격+시그니처메뉴
foodtype_menu_acc = results[2]['정확도']  # 음식종류+시그니처메뉴
price_foodtype_acc = results[3]['정확도']  # 평균가격+음식종류
restaurant_acc = results[4]['정확도']  # 식당명 포함
region_acc = results[5]['정확도']  # 지역 포함
rating_acc = results[6]['정확도']  # 평점 포함

print("\n" + "="*100)
print("🔍 기여도 분석")
print("="*100)

print(f"\n📌 Baseline (음식종류+시그니처메뉴+평균가격): {baseline_acc:.2f}%")

# 음식 종류 기여도(A) = Baseline - [평균 가격 + 시그니처메뉴] 정확도
A = baseline_acc - price_menu_acc
print(f"\n🍜 음식 종류 기여도(A):")
print(f"   Baseline 정확도: {baseline_acc:.2f}%")
print(f"   [평균가격+시그니처메뉴] 정확도: {price_menu_acc:.2f}%")
print(f"   기여도(A): {A:+.2f}%p")

# 평균 가격 기여도(B) = Baseline - [음식 종류 + 시그니처메뉴] 정확도
B = baseline_acc - foodtype_menu_acc
print(f"\n💰 평균 가격 기여도(B):")
print(f"   Baseline 정확도: {baseline_acc:.2f}%")
print(f"   [음식종류+시그니처메뉴] 정확도: {foodtype_menu_acc:.2f}%")
print(f"   기여도(B): {B:+.2f}%p")

# 시그니처메뉴 기여도(C) = Baseline - [평균 가격 + 음식 종류] 정확도
C = baseline_acc - price_foodtype_acc
print(f"\n🍽️  시그니처메뉴 기여도(C):")
print(f"   Baseline 정확도: {baseline_acc:.2f}%")
print(f"   [평균가격+음식종류] 정확도: {price_foodtype_acc:.2f}%")
print(f"   기여도(C): {C:+.2f}%p")

# 식당명 기여도(D) = 식당명 포함 후 정확도 - Baseline
D = restaurant_acc - baseline_acc
print(f"\n🏪 식당명 기여도(D):")
print(f"   식당명 포함 정확도: {restaurant_acc:.2f}%")
print(f"   Baseline 정확도: {baseline_acc:.2f}%")
print(f"   기여도(D): {D:+.2f}%p")

# 지역 기여도(E) = 지역 포함 후 정확도 - Baseline
E = region_acc - baseline_acc
print(f"\n📍 지역 기여도(E):")
print(f"   지역 포함 정확도: {region_acc:.2f}%")
print(f"   Baseline 정확도: {baseline_acc:.2f}%")
print(f"   기여도(E): {E:+.2f}%p")

# 평점 기여도(F) = 평점 포함 후 정확도 - Baseline
F = rating_acc - baseline_acc
print(f"\n⭐ 평점 기여도(F):")
print(f"   평점 포함 정확도: {rating_acc:.2f}%")
print(f"   Baseline 정확도: {baseline_acc:.2f}%")
print(f"   기여도(F): {F:+.2f}%p")

# 각 기여도의 가중치 계산
total_contribution = A + B + C + D + E + F
print(f"\n" + "="*100)
print("📊 가중치 계산")
print("="*100)
print(f"\n총 기여도 합: {total_contribution:.2f}%p")

if abs(total_contribution) > 0.0001:  # 0이 아닌 경우에만 계산
    weight_A = A / total_contribution
    weight_B = B / total_contribution
    weight_C = C / total_contribution
    weight_D = D / total_contribution
    weight_E = E / total_contribution
    weight_F = F / total_contribution
    
    print(f"\n🎯 각 특성의 가중치:")
    print(f"   음식 종류 가중치: {weight_A:.4f} ({weight_A*100:.2f}%)")
    print(f"   평균 가격 가중치: {weight_B:.4f} ({weight_B*100:.2f}%)")
    print(f"   시그니처메뉴 가중치: {weight_C:.4f} ({weight_C*100:.2f}%)")
    print(f"   식당명 가중치: {weight_D:.4f} ({weight_D*100:.2f}%)")
    print(f"   지역 가중치: {weight_E:.4f} ({weight_E*100:.2f}%)")
    print(f"   평점 가중치: {weight_F:.4f} ({weight_F*100:.2f}%)")
    
    # 가중치 합 검증
    total_weight = weight_A + weight_B + weight_C + weight_D + weight_E + weight_F
    print(f"\n   가중치 합: {total_weight:.4f} (검증)")
else:
    print("\n⚠️  총 기여도 합이 0에 가까워 가중치를 계산할 수 없습니다.")
    weight_A = weight_B = weight_C = weight_D = weight_E = weight_F = 0

# 7. 결과 저장
print("\n💾 결과 저장 중...")

# 상세 결과
detailed_results = []
for i, exp in enumerate(experiments):
    detailed_results.append({
        '실험번호': i + 1,
        '실험명': exp['name'],
        '정확도(%)': f"{results[i]['정확도']:.2f}",
        '정확개수': results[i]['정확개수'],
        '전체개수': results[i]['전체개수'],
        'Baseline 대비 변화': f"{results[i]['정확도'] - baseline_acc:+.2f}%p"
    })

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv("ablation_study_results.csv", index=False, encoding='utf-8-sig')
print("✅ 상세 결과 저장: ablation_study_results.csv")

# 기여도 및 가중치 요약
if abs(total_contribution) > 0.0001:
    contribution_summary = {
        'Baseline 정확도': f"{baseline_acc:.2f}%",
        '음식종류 기여도(A)': f"{A:+.2f}%p",
        '평균가격 기여도(B)': f"{B:+.2f}%p",
        '시그니처메뉴 기여도(C)': f"{C:+.2f}%p",
        '식당명 기여도(D)': f"{D:+.2f}%p",
        '지역 기여도(E)': f"{E:+.2f}%p",
        '평점 기여도(F)': f"{F:+.2f}%p",
        '총 기여도 합': f"{total_contribution:.2f}%p",
        '음식종류 가중치': f"{weight_A:.4f} ({weight_A*100:.2f}%)",
        '평균가격 가중치': f"{weight_B:.4f} ({weight_B*100:.2f}%)",
        '시그니처메뉴 가중치': f"{weight_C:.4f} ({weight_C*100:.2f}%)",
        '식당명 가중치': f"{weight_D:.4f} ({weight_D*100:.2f}%)",
        '지역 가중치': f"{weight_E:.4f} ({weight_E*100:.2f}%)",
        '평점 가중치': f"{weight_F:.4f} ({weight_F*100:.2f}%)",
        '최고 정확도': f"{max([r['정확도'] for r in results]):.2f}%",
        '최고 정확도 실험': [r['실험명'] for r in results if r['정확도'] == max([r['정확도'] for r in results])][0]
    }
else:
    contribution_summary = {
        'Baseline 정확도': f"{baseline_acc:.2f}%",
        '음식종류 기여도(A)': f"{A:+.2f}%p",
        '평균가격 기여도(B)': f"{B:+.2f}%p",
        '시그니처메뉴 기여도(C)': f"{C:+.2f}%p",
        '식당명 기여도(D)': f"{D:+.2f}%p",
        '지역 기여도(E)': f"{E:+.2f}%p",
        '평점 기여도(F)': f"{F:+.2f}%p",
        '총 기여도 합': f"{total_contribution:.2f}%p",
        '음식종류 가중치': "계산 불가",
        '평균가격 가중치': "계산 불가",
        '시그니처메뉴 가중치': "계산 불가",
        '식당명 가중치': "계산 불가",
        '지역 가중치': "계산 불가",
        '평점 가중치': "계산 불가",
        '최고 정확도': f"{max([r['정확도'] for r in results]):.2f}%",
        '최고 정확도 실험': [r['실험명'] for r in results if r['정확도'] == max([r['정확도'] for r in results])][0]
    }

summary_df = pd.DataFrame([contribution_summary])
summary_df.to_csv("ablation_contribution_summary.csv", index=False, encoding='utf-8-sig')
print("✅ 기여도 및 가중치 요약 저장: ablation_contribution_summary.csv")

# 8. 시각화 (텍스트 기반)
print("\n" + "="*100)
print("📈 정확도 비교 차트")
print("="*100)

max_acc = max([r['정확도'] for r in results])
for i, result in enumerate(results):
    bar_length = int((result['정확도'] / max_acc) * 50)
    bar = "█" * bar_length
    change = result['정확도'] - baseline_acc
    change_str = f"({change:+.2f}%p)" if i > 0 else ""
    print(f"{result['실험명']:30s}: {result['정확도']:6.2f}% {change_str:15s} {bar}")

print("\n" + "="*100)
print("🎉 Ablation Study 완료!")
print("="*100)

print(f"\n✨ 핵심 발견:")
contributions = [
    ('음식 종류', A),
    ('평균 가격', B),
    ('시그니처메뉴', C),
    ('식당명', D),
    ('지역', E),
    ('평점', F)
]
sorted_contributions = sorted(contributions, key=lambda x: x[1], reverse=True)

print(f"\n   기여도 순위:")
for i, (name, contrib) in enumerate(sorted_contributions, 1):
    print(f"   {i}. {name}: {contrib:+.2f}%p")

if abs(total_contribution) > 0.0001:
    weights_list = [
        ('음식 종류', weight_A),
        ('평균 가격', weight_B),
        ('시그니처메뉴', weight_C),
        ('식당명', weight_D),
        ('지역', weight_E),
        ('평점', weight_F)
    ]
    sorted_weights = sorted(weights_list, key=lambda x: x[1], reverse=True)
    print(f"\n   가중치 순위:")
    for i, (name, weight) in enumerate(sorted_weights, 1):
        print(f"   {i}. {name}: {weight:.4f} ({weight*100:.2f}%)")

best_exp = max(results, key=lambda x: x['정확도'])
print(f"\n🏆 최고 성능: {best_exp['실험명']} ({best_exp['정확도']:.2f}%)")