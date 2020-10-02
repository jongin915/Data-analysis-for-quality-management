import numpy as np
import pandas as pd

data = pd.read_csv("EAGLE_P5(Raw).csv", sep=',')
train_data= data.iloc[:700,:]
a=train_data.corr()
a=np.array(a)
variable_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
b=[]
for i in range(0,14):
    b.append(abs(a[i][-1]))
np.argmax(b) #값8
variable_list[8]#값 8, X9을 선택
print("첫번째 변수 = X%d"%(variable_list[8]+1))
variable_list.remove(variable_list[8])

c=[]
for i in range(0,14):
    c.append(abs(a[i][-1]) - abs(a[i][8]))
c.remove(c[8])
np.argmax(c) #값7
variable_list[7] #값 7 X8을 선택
print("두번째 변수 = X%d"%(variable_list[7]+1))
variable_list.remove(variable_list[7])

d=[]
for i in range(0,14):
    d.append(abs(a[i][-1])-(abs(a[i][7])+abs(a[i][8]))/2)
d.remove(d[8])
d.remove(d[7])
np.argmax(d) #값10
variable_list[10] #값12 X13을 선택
print("세번째 변수 = X%d"%(variable_list[10]+1))
variable_list.remove(variable_list[10])

e=[]
for i in range(0,14):
    e.append(abs(a[i][-1])-(abs(a[i][7])+abs(a[i][8])+abs(a[i][12]))/3)
e.remove(e[12])
e.remove(e[8])
e.remove(e[7])
np.argmax(e) #값5
variable_list[5] #값5 X6을 선택
print("네번째 변수 = X%d"%(variable_list[5]+1))
variable_list.remove(variable_list[5])

f=[]
for i in range(0,14):
    f.append(abs(a[i][-1])-(abs(a[i][5])+abs(a[i][7])+abs(a[i][8])+abs(a[i][12]))/4)
f.remove(f[12])
f.remove(f[8])
f.remove(f[7])
f.remove(f[5])
np.argmax(f) #값 9
variable_list[9] #값13 X14를 선택
print("다섯번째 변수 = X%d"%(variable_list[9]+1))
variable_list.remove(variable_list[9])