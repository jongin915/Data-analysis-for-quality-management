import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix #혼동 행렬을 위해


data = pd.read_csv('wine(Range).csv')
trainX = data.iloc[:1119, 0:-1]
trainY = data.iloc[:1119, -1]
testX = data.iloc[1119:,0:-1]
testY = data.iloc[1119:,-1]

LR = LogisticRegression(penalty="l1",solver='liblinear')#penalty l2면 그냥 LR, l1이면 Lasso
LR.fit(trainX, trainY)
Y_hat_train = LR.predict(trainX)
Y_hat = LR.predict(testX)
CM=confusion_matrix(testY,Y_hat)
Precision=float(CM[1,1]/(CM[1,1]+CM[0,1]))
Recall=float(CM[1,1]/(CM[1,1]+CM[1,0]))
Specificity=float(CM[0,0]/(CM[0,0]+CM[0,1]))
F1measure=float(2*Precision*Recall/(Precision+Recall))
G_mean=float(np.sqrt(Precision*Recall))

print("Confusion Matrix_test:")
print(CM)
print("Precision_test:%.3f, Recall_test:%.3f, Specificity_test:%.3f, F1measure_test:%.3f, G-mean_test:%.3f" %(Precision, Recall, Specificity, F1measure, G_mean))


print ("Accuracy, training set:%.3f"%(LR.score(trainX, trainY)))
print ("Accuracy, test set:%.3f"%(LR.score(testX, testY)))