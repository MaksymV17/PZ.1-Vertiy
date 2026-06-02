from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

y_true = [1,0,1,1,0,1]
y_pred = [1,0,1,0,0,1]

print(confusion_matrix(y_true,y_pred))
print("Accuracy:",accuracy_score(y_true,y_pred))
print("Precision:",precision_score(y_true,y_pred))
print("Recall:",recall_score(y_true,y_pred))
print("F1:",f1_score(y_true,y_pred))