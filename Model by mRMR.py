from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np


# 파라미터 설정
modelName = Ridge #XGBRegressor, Lasso, ElasticNet

train_rate = 700/900

# 1. 데이터 불러오기
rawData = np.genfromtxt("EAGLE_P5(Raw)(mRMR).csv", delimiter=",", skip_header=1)
ndata = rawData.shape[0]
ncolumn = rawData.shape[1]

rawData[0,:] #first row data
rawData[:,(ncolumn-1)] #last column data

# 2. train, test 데이터 정의
ntrain = int(ndata * train_rate)
train_index = range(ntrain)
test_index = range(ntrain, ndata)
train, test = rawData[train_index,], rawData[test_index,]
train_x, train_y = train[:,:(ncolumn-1)], train[:,(ncolumn-1)]
test_x, test_y = test[:,:(ncolumn-1)], test[:,(ncolumn-1)]

model = modelName()

# 4. 모델 학습
model.fit(train_x, train_y)

# 5. 성능 평가 결과 확인
train_perf = r2_score(train_y, model.predict(train_x))
test_perf = r2_score(test_y, model.predict(test_x))
MSE = mean_squared_error(model.predict(test_x), test_y)

print("학습 데이터로 측정한 모델의 성능은 %.4f 입니다." %(train_perf))
print("테스트 데이터로 측정한 모델의 성능은 %.4f 입니다." %(test_perf))
print("테스트 데이터에 대한 MSE 값은 %.4f 입니다." %(MSE))