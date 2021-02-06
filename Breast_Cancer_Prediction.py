import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("data.csv")
print(data.head())
#data = data.drop(columns=["Unnamed:32"])
print(data.sample(5))
MGT = data[data["diagnosis"]=="M"].sample(5)
BGN = data[data["diagnosis"]=="B"].sample(5)
M = MGT.iloc[:,2:].values
B = BGN.iloc[:,2:].values
# euclidian distances between 5 random malignant and benign tumors
((M-B)**2).sum(axis=1)**0.5
MGT1 = data[data["diagnosis"]=="M"].sample(5)
MGT2 = data[data["diagnosis"]=="M"].sample(5)
M1 = MGT1.iloc[:,2:].values
M2 = MGT2.iloc[:,2:].values
# euclidian distancec between 5 random malignant tumors
print(((M1-M2)**2).sum(axis=1)**0.5)
BGN1 = data[data["diagnosis"]=="B"].sample(5)
BGN2 = data[data["diagnosis"]=="B"].sample(5)
B1 = BGN1.iloc[:,2:].values
B2 = BGN2.iloc[:,2:].values
print(((B1-B2)**2).sum(axis=1)**0.5)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
d = {"M":1,"B":0}
data["diagnosis"] = data["diagnosis"].map(d)
print(data)
Y = data["diagnosis"]
X = data.iloc[:,2:]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)
model = KNeighborsClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_test.values
print(y_pred)
(y_test.values == y_pred).mean()
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
print(confusion_matrix(y_test.values , y_pred))
print(precision_recall_fscore_support(y_test.values , y_pred, average="weighted"))
# hyper parameter tuning
k = []
accuracy = []
for i in range(2,100):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    k.append(i)
    accuracy.append((y_pred==y_test.values).mean())
plt.plot(k,accuracy)
plt.title("K value vs accuracy")
plt.xlabel("K - value")
plt.ylabel("accuracy")
plt.show()
model = KNeighborsClassifier(n_neighbors=20)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
(y_test.values == y_pred).mean()
print(confusion_matrix(y_test.values , y_pred))
print(precision_recall_fscore_support(y_test.values , y_pred, average="weighted"))