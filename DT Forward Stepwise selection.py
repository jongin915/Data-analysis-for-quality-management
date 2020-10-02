import sklearn
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix #혼동 행렬을 위해
import numpy as np


data = pd.read_csv('wine(Range)(1,3,5,8,9,10).csv')
trainX = data.iloc[:1119, 0:-1]
trainY = data.iloc[:1119, -1]
testX = data.iloc[1119:,0:-1]
testY = data.iloc[1119:,-1]

DT = DecisionTreeClassifier(criterion = 'entropy',random_state=1,max_depth=5, min_samples_split=30) #random state 지정 안하면 값이 계속 바뀜
DT = DT.fit(trainX, trainY) #max_depth 17일 때부터 똑같이 나옴
Y_hat = DT.predict(testX)

CM=confusion_matrix(testY,Y_hat)
Precision=float(CM[1,1]/(CM[1,1]+CM[0,1]))
Recall=float(CM[1,1]/(CM[1,1]+CM[1,0]))
Specificity=float(CM[0,0]/(CM[0,0]+CM[0,1]))
F1measure=float(2*Precision*Recall/(Precision+Recall))
G_mean=float(np.sqrt(Precision*Recall))

print("Confusion Matrix_test:")
print(CM)
print("Precision_test:%.3f, Recall_test:%.3f, Specificity_test:%.3f, F1measure_test:%.3f, G-mean_test:%.3f" %(Precision, Recall, Specificity, F1measure, G_mean))

Y_hat_train = DT.predict(trainX)
CM_train=confusion_matrix(trainY,Y_hat_train)
Precision_t=float(CM_train[1,1]/(CM_train[1,1]+CM_train[0,1]))
Recall_t=float(CM_train[1,1]/(CM_train[1,1]+CM_train[1,0]))
Specificity_t=float(CM_train[0,0]/(CM_train[0,0]+CM_train[0,1]))
F1measure_t=float(2*Precision_t*Recall_t/(Precision_t+Recall_t))
G_mean_t=float(np.sqrt(Precision_t*Recall_t))


print ("Accuracy, training set:%.3f"%(DT.score(trainX, trainY)))
print ("Accuracy, test set:%.3f"%(DT.score(testX, testY)))