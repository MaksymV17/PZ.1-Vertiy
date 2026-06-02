from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score

X = np.array([[1,2],[2,3],[3,1],[6,5],[7,8],[8,6]])
y = np.array([0,0,0,1,1,1])

model = svm.SVC()
model.fit(X,y)

pred = model.predict(X)

print(pred)
print("Accuracy:",accuracy_score(y,pred))