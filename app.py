# 라이브러리 임포트
import streamlit as st
import numpy as np
import pandas as pd
from datasets import load_dataset
import platform
from matplotlib import rc
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from operator import itemgetter

# 한글 폰트 사용토록 글꼴 설정 변경
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic') 
elif platform.system() == 'Windows':
    rc('font', family='Malgun Gothic') 
elif platform.system() == 'Linux':
    rc('font', family='Malgun Gothic') 
# 축 레이블에서 '-' 기호 깨지지 않기 위한 설정
plt.rcParams['axes.unicode_minus'] = False

# 상수 정의
# 로컬 파일 경로
data_path = "./dataset/"
# Hugging Face 저장소 정보
repo_id = f"{'developzest'}/{'kaggle_instacart'}"

scaler = StandardScaler()

# 로컬 데이터 로드
@st.cache_data
def load_local_data():
    train = pd.read_csv(data_path + 'order_products__train.csv')
    orders_train_test = pd.read_csv(filepath_or_buffer=data_path + 'orders_train_test.csv', low_memory=False)
    
    products = pd.read_csv(data_path + 'products.csv')
    aisles = pd.read_csv(data_path + 'aisles.csv')
    departments = pd.read_csv(data_path + 'departments.csv')
    categorized_products = pd.merge(left=products, right=aisles, how='inner', on='aisle_id')
    categorized_products = pd.merge(left=categorized_products, right=departments, how='inner', on='department_id')
    categorized_products.drop(columns=['aisle_id', 'department_id'], inplace=True)
    
    return train.copy(), orders_train_test.copy(), categorized_products.copy()

# 허깅페이스 데이터 로드
@st.cache_data
def load_huggingface_data():
    uxp = load_dataset(repo_id, revision="v1.0")['uxp'].to_pandas()        
    uxp['one_shot_ratio_product'] = uxp['one_shot_ratio_product'].fillna(value=0)
    uxp['times_last5'] = uxp['times_last5'].fillna(value=0)
    uxp['times_last5_ratio'] = uxp['times_last5_ratio'].fillna(value=0)
    return uxp.copy()

# 모델용 데이터 생성
@st.cache_data
def get_model_train_test_data(uxp, train, orders_train_test):
    merged_uxp_train_test_orders = pd.merge(left=uxp, right=orders_train_test, on='user_id', how='left')
    uxp_train = pd.merge(left=merged_uxp_train_test_orders.loc[merged_uxp_train_test_orders['eval_set'] == 'train'], 
                         right=train, 
                         on=['product_id', 'order_id'], 
                         how='left').drop(columns=['order_id','eval_set', 'add_to_cart_order']).set_index(['user_id', 'product_id'])
    uxp_train['reordered'] = uxp_train['reordered'].fillna(0)
    
    uxp_test = merged_uxp_train_test_orders.loc[merged_uxp_train_test_orders['eval_set'] == 'test'].drop(columns=['eval_set', 'order_id']).set_index(['user_id', 'product_id'])
    return uxp_train.copy(), uxp_test.copy()

# 모델 학습 함수
@st.cache_resource
def train_model(uxp_train):
    X = uxp_train.drop(columns=['reordered'])
    y = uxp_train['reordered']

    X_scaled = scaler.fit_transform(X)
    
    params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.76,
    'bagging_fraction': 0.75,
    'bagging_freq': 5
    }
    ROUNDS = 50
    
    model = lgb.train(params, lgb.Dataset(X_scaled, label=y), ROUNDS)
    return model

class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)

def get_best_prediction(items, preds, pNone=None):
    items_preds = sorted(list(zip(items, preds)), key=itemgetter(1), reverse=True)
    P = [p for i,p in items_preds]
    L = [i for i,p in items_preds]
    
    opt = F1Optimizer.maximize_expectation(P)
    best_prediction = []
    best_prediction += (L[:opt[0]])
    if best_prediction == []:
        best_prediction = ['None']
    return ' '.join(list(map(str,best_prediction)))

# def multi(sub, i):
#     items = sub.loc[i,'product_id']
#     preds = sub.loc[i,'yhat']
#     ret = get_best_prediction(items, preds)
#     return ret

# 상품 예측 함수
def predict_products(model, uxp_test, orders_train_test):
    X_test_scaled = scaler.fit_transform(uxp_test)
    
    uxp_test["reordered"] = model.predict(X_test_scaled)
    uxp_test = uxp_test.reset_index()
    uxp_test = uxp_test[['product_id', 'user_id', "reordered"]]
    st.write("모델 예측 데이터 추출")
    st.dataframe(uxp_test.head(10))
    
    uxp_test = uxp_test.merge(orders_train_test.loc[orders_train_test['eval_set'] == 'test', ["user_id", "order_id"]], on='user_id', how='left').drop('user_id', axis=1)
    uxp_test.columns = ['product_id', 'prediction', 'order_id']
    uxp_test.product_id = uxp_test.product_id.astype(int)
    uxp_test.order_id = uxp_test.order_id.astype(int)
    
    sub_item = uxp_test.groupby(['order_id','product_id']).prediction.mean().reset_index()
    sub = sub_item.groupby('order_id').product_id.apply(list).to_frame()
    sub['yhat'] = sub_item.groupby('order_id').prediction.apply(list)
    sub.reset_index(inplace=True)
    
    st.write('최적 예측 위한 연산용 데이터')
    st.dataframe(sub.head(10))
    
    with st.spinner("최적 데이터 연산 중으로 시간이 오래 결립니다..."):
        sub['products'] = sub.loc[:, ['product_id', 'yhat']].apply(lambda x: get_best_prediction(x[0], x[1]), axis=1, raw=True)    
        sub.reset_index(inplace=True)
        sub = sub[['order_id', 'products']]
        return sub

# Streamlit 애플리케이션
st.title("고객 주문 제품 예측 애플리케이션")

# 로컬 데이터 로드
if st.sidebar.button("로컬 데이터 로드"):
    with st.spinner("데이터를 로드 중입니다..."):
        train, orders_train_test, categorized_products = load_local_data()
        st.session_state.train = train
        st.session_state.orders_train_test = orders_train_test
        
        st.write('훈련용 데이터에 맵핑되는 주문 데이터')
        st.dataframe(train.sample(5))
        st.write('train, test에 속한 고객 주문 데이터')
        st.dataframe(orders_train_test.sample(5))
        st.write('제품의 종류 정보가 포함된 데이터')
        st.dataframe(categorized_products.sample(5))
        
        st.success("로컬 데이터 로드 완료!")

# Feature Engineering 데이터 로드
if st.sidebar.button("Huggingface 데이터 로드"):
    with st.spinner("데이터를 로드 중입니다..."):
        uxp = load_huggingface_data()
        st.session_state.uxp = uxp
        
        st.write('Feature Engineering한 고객x상품 데이터')
        st.dataframe(uxp.sample(5))
        
        st.success("Huggingface 데이터 로드 완료!")

# 모델 학습
if st.sidebar.button("모델용 데이터 로드 및 모델 학습"):
    if ('train' not in st.session_state) or ('orders_train_test' not in st.session_state):
        st.error("먼저 로컬 데이터를 로드해주세요.", icon="🚨")
    elif 'uxp' not in st.session_state:
        st.error("먼저 Huggingface 데이터를 로드해주세요.", icon="🚨")
    else:
        with st.spinner("모델 학습 및 예측 용 데이터를 로드 중입니다..."):
            uxp_train, uxp_test = get_model_train_test_data(st.session_state.uxp, st.session_state.train, st.session_state.orders_train_test)
            st.session_state.uxp_test = uxp_test
            
            st.write(f"모델 train 데이터")
            st.dataframe(uxp_train.sample(5))
            st.write(f"모델 test 데이터")
            st.dataframe(uxp_test.sample(5))
            
            with st.spinner("모델을 학습 중입니다... 잠시만 기다려주세요."):
                model = train_model(uxp_train)
                st.session_state.model = model
                
                st.pyplot(lgb.plot_importance(model).figure)
                st.success("모델 학습 완료!")

# 고객 주문 제품 예측 및 submission 파일 다운로드 버튼 표시
if st.sidebar.button("고객 주문 제품 예측"):
    if 'model' in st.session_state:
        with st.spinner("고객 주문 제품을 예측 중입니다..."):
            feature_data = predict_products(st.session_state.model, st.session_state.uxp_test, st.session_state.orders_train_test)
            st.session_state.feature_data = feature_data
            
            st.write("Instacart products per order prediction")
            st.dataframe(feature_data.head())
            st.success("모델 예측 완료!")
            
            st.sidebar.download_button("Press to Download", st.session_state.feature_data.to_csv(index=False).encode('utf-8'), "submission.csv", "text/csv", key='download-csv')
    else:
        st.error("먼저 모델을 학습시켜주세요.", icon="🚨")