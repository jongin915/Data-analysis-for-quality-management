from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

modelName = Ridge


data = pd.read_csv("EAGLE_P5(Raw).csv")
trainX = data.iloc[:700, 0:-1]
trainY = data.iloc[:700, -1]
testX = data.iloc[700:,0:-1]
testY = data.iloc[700:,-1]

trans = trainX.T
covariance_matrix = np.cov(trans)
eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
rate = np.round(np.cumsum(eig_vals*100)/sum(eig_vals),decimals=3)
print(rate) #k=9일때 가장높음

pca = PCA(n_components=9)
newx = pca.fit_transform(trainX)
newx_test = pca.transform(testX)

model = modelName(alpha=6)


# 4. 모델 학습
model.fit(newx, trainY)
MSE = mean_squared_error(model.predict(newx_test), testY)

# 5. 성능 평가 결과 확인
train_perf = r2_score(trainY, model.predict(newx))
test_perf = r2_score(testY, model.predict(newx_test))
print("학습 데이터로 측정한 모델의 성능은 %.4f 입니다." %(train_perf))
print("테스트 데이터로 측정한 모델의 성능은 %.4f 입니다." %(test_perf))
print("테스트 데이터에 대한 MSE 값은 %.4f 입니다." %(MSE))