# ML Project for Kaggle Instacart 

# 1. 협업방식
## 1-1. 실제사용한 깃헙 컨벤션
- 커밋 메시지 양식
    > '[타입]: <제목>'
- [타입] 참고사항
    - `Feat` : 기능 (새로운 기능)
    - `Update` : 기능 업데이트
    - `Fix` : 에러 수정
    - `Del` : 삭제
    - `Refactor` : 리팩토링
    - `Docs` : 문서 (문서 추가, 수정, 삭제)
    - `Chore` : 잡다한 일
## 1-2. 데이터셋 관리방법
- huggingface 로 데이터셋 형상관리
- [huggingface repo url](https://huggingface.co/datasets/developzest/kaggle_instacart/commit/f8adf4227a855448d5a92c47bb951b698f4cd517)
## 1-3. 의사소통 및 실험기록 방법 등
- 매일 오후 4시 zoom
---

# 2. 대회 및 데이터 개요
## 2.1. 대회 설명
- 시간이 지남에 따라 고객 주문에 대한 익명화된 데이터를 사용하여 이전에 구매한 제품이 사용자의 다음 주문에 있을지 예측하는 최고의 모델을 찾는 대회입니다.
즉, **대회의 목표는 어떤 제품이 사용자의 다음 주문에 있을지 예측하는 것**입니다.
- [Kaggle Competition Page](https://www.kaggle.com/competitions/instacart-market-basket-analysis)

## 2.2. 데이터 설명
- 이 데이터셋은 **시간**에 따른 **고객 주문**을 설명하는 csv 파일들로 구성되어 있습니다.
- 이 데이터는 사용자가 **어떤 물품을 다시 주문할지 예측**하는 문제를 풀어야 합니다.
- 데이터셋은 **익명화**되어 있으며 20만 명이 넘는 instacart 사용자의 300만건 이상의 주문들을 포함하고 있습니다.
- 각 사용자별로, 4회에서 100회 사이의 주문 데이터가 제공됩니다. 또한 주문이 이루어진 주와 시간, 그리고 주문 간의 상대적인 시간을 제공합니다.
- 데이터 파일은 6개로 구성되어 있으며, 각 csv 파일에 대한 설명은 다음과 같습니다.

### [csv별 설명]
1. orders.csv
    - order_id: 주문 식별자
    - user_id: 고객 식별자
    - eval_set: 주문이 속한 evaluation set
    - order_number: 고객의 주문 번호 (1 = 첫 번째, n = n번째)
    - order_dow: 주문한 요일
    - order_hour_of_day: 주문이 이루어진 날의 시간
    - days_since_prior: 마지막 주문 이후 일수, 30일로 제한(주문 번호 = 1의 경우 NA)
2. products.csv
    - product_id: 제품 식별자
    - product_name: 제품명
    - aisle_id: 제품 중분류 식별자
    - department_id: 제품 대분류 식별자
3. aisles.csv
    - aisle_id: 제품 중분류 식별자
    - aisle: 제품 중분류 명
4. deptartments.csv
    - department_id: 제품 대분류 식별자
    - department: 제품 대분류 명
5. order_products__SET.csv
    - order_id: 주문 식별자
    - product_id: 제품 식별자
    - add_to_cart_order:  각 제품이 장바구니에 추가된 순서
    - reordered: 제품이 과거에 해당 고객에 의해 주문된 경우 1, 그렇지 않으면 0
    - 여기서 SET 은 orders.csv의 eval_set 중 하나
        - "prior": 고객이 이전에 주문한 가장 최근 주문 데이터
        - "train": 대회를 위해 참가자에게 제공된 훈련 데이터
        - "test":  대회를 위해 참가자에게 제공된 테스트 데이터
---

# 3. 데이터의 이해
## 3.1 EDA를 통한 인사이트
- Instacart에서 고객이 구매한 상품의 대분류 비율
- 재구매율이 높았던 상품
- 카트에 가장 먼저 담은 상품
- etc...
## 3.2 데이터 전처리 근거와 방법
- XGBoost와 LightGBM은 내부적으로 결측치를 처리하나 결측 데이터를 일괄적으로 0으로 처리
- 고유값을 갖는 feature들이 존재하므로 정규화 진행
## 3.3 Feature Engineering 근거와 방법
- 문제 해결을 위해 고객의 이전 구매 패턴과 제품 정보를 알아야 하기 때문에 전체 주문 정보와 고객이 이전에 주문한 메타 정보 결합하는 방식으로 진행
    - train과 test에 속한 고객의 이전 주문 정보가 order_products__prior.csv에 있으므로 이를 활용하여 추가 feature 생성
- feature engineering 전 분류 모델 별 feature importance 를 '[TEST]_ML_Project_Kaggle_Instacart_Market_Basket_Analysis.ipynb'에서 확인하여 해당 feature 활용
- '[FE]_ML_Project_Kaggle_Instacart.ipynb' 에서 feature engineering 진행
---

# 4. 모델링
## 4.1 선택한 모델과 선택한 이유
- 선택 모델 : LightGBM
- 선택 이유
    - 1. Gradient가 큰 데이터는 유지하고 Gradient가 낮은 데이터는 Randomly Drop을 수행
    - 2. Feature Engineering 전 '[TEST]_ML_Project_Kaggle_Instacart_Market_Basket_Analysis.ipynb'에서 
    여러 분류 모델 및 앙상블 모델을 통해 테스트한 결과 성능 평가가 가장 높은 LightGBM 선택
## 4.2 Hyper Parameter Tuning 방법
- GridSearch 및 중첩반복문 사용시, 실행시간이 너무 오래 걸려 각 파라미터별로 확인
---

# 5. 평가
## 5.1 해당 대회의 평가 산식과 실제 대회를 진행한 결과에서 나온 인사이트를 작성해주세요.
- 대회 평가 산식 : mean f1 score
- 인사이트 : 해당 대회 리더보드 1 ~ 3등의 해결책으로 CatBoost와 Word2Vec을 사용했다고 한 것을 보아 원본 데이터에 대해 추가적인 분류 작업이 더 필요해보임 

