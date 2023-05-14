import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score , confusion_matrix

data1 = pd.read_csv('Epileptic Seizure Recognition.csv')
data2 = pd.read_csv('Epileptic Seizure Recognition2.csv')
# print(ESR.head())
cols = data1.columns
tgt =data1.y
#print(tgt)##################################
tgt[tgt>1] = 0

x = data1.iloc[: , 1 : 179 ]
# print(x.head(3))##############################
# print(x.shape)

y= data1.iloc[:,179]
# print(y.head(3))#############################
# print(y)
# y[y>1]=0
# print(y)
##########################################################
#------------- //  Split / train // test  \\ -------------
##########################################################
from sklearn.model_selection import train_test_split, cross_val_score
X_train,X_test ,y_train , y_test = train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

##########################################################
#------------------ // classifiers \\--------------
##########################################################

# #-------------  LogisticRegression -----------
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression()

clf.fit(X_train,y_train)

y_pred_log_reg = clf.predict(X_test)

acc_log_reg = accuracy_score(y_test,y_pred_log_reg)*100

print("LogisticRegression = ",str(acc_log_reg) + '%')

cm = confusion_matrix(y_test, y_pred_log_reg)

print(cm)

# #----------- SVC -------------------
from sklearn.svm import SVC

clf=SVC()

clf.fit(X_train,y_train)

y_pred_svc = clf.predict(X_test)

acc_svc = accuracy_score(y_test,y_pred_svc)*100

print("Default SVC = ",str(acc_svc) + '%')

cm = confusion_matrix(y_test, y_pred_svc)

print(cm)

clf=SVC(kernel='sigmoid', C=2)

clf.fit(X_train,y_train)

y_pred_svc = clf.predict(X_test)

acc_svc = accuracy_score(y_test,y_pred_svc)*100

print("Custom SVC = ", str(acc_svc) + '%')

cm = confusion_matrix(y_test, y_pred_svc)

print(cm)

# #--------------- Linear SVC-----------
from sklearn.svm import SVC , LinearSVC

clf = LinearSVC()

clf.fit(X_train,y_train)

y_pred_linearSVC = clf.predict(X_test)

acc_linear_svc = accuracy_score(y_test,y_pred_linearSVC)*100

print("default SVC linear = " , str(acc_linear_svc) + '%')

cm = confusion_matrix(y_test, y_pred_linearSVC)

print(cm)

clf = LinearSVC(dual=False)

clf.fit(X_train,y_train)

y_pred_linearSVC = clf.predict(X_test)

acc_linear_svc = accuracy_score(y_test,y_pred_linearSVC)*100

print("custom SVC linear = " , str(acc_linear_svc) + '%')

cm = confusion_matrix(y_test, y_pred_linearSVC)

print(cm)

# ---------- KNN ----------------
from sklearn.neighbors import KNeighborsClassifier
#default K = 5
clf=KNeighborsClassifier() 

clf.fit(X_train,y_train)

y_pred_knn = clf.predict(X_test)

#acc_knn = round(clf.score(y_test,y_pred_knn) * 100 , 2)
acc_knn=accuracy_score(y_test,y_pred_knn)*100

#print(zz)
#print('KNN = ' , str(acc_knn) + '%')

print('default KNN = ' , str(acc_knn) + '%')

cm = confusion_matrix(y_test, y_pred_knn)

print(cm)

clf=KNeighborsClassifier(n_neighbors=3) 

clf.fit(X_train,y_train)

y_pred_knn = clf.predict(X_test)

acc_knn=accuracy_score(y_test,y_pred_knn)*100

print('custom KNN = ' , str(acc_knn) + '%')

cm = confusion_matrix(y_test, y_pred_knn)

print(cm)

# # ----------  Decision  Tree   ----------------
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

clf = DecisionTreeClassifier()

clf=clf.fit(X_train,y_train)

y_Pred = clf.predict(X_test)

acc_dec= accuracy_score(y_test,y_Pred)*100

print("default Decision Tree accuracy :", acc_dec , '%')

cm = confusion_matrix(y_test, y_Pred)

print(cm)

clf = DecisionTreeClassifier(splitter='random')

clf=clf.fit(X_train,y_train)

y_Pred = clf.predict(X_test)

acc_dec= accuracy_score(y_test,y_Pred)*100

print("Custom Decision Tree accuracy :", acc_dec , '%')

cm = confusion_matrix(y_test, y_Pred)

print(cm)

# #--------------- GaussianNB --------------------
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(X_train,y_train)

y_pred_gnb = clf.predict(X_test)

acc_gnb = accuracy_score(y_test,y_pred_gnb)*100

print("Gaussian = " , str(acc_gnb) + '%')
cm = confusion_matrix(y_test, y_pred_gnb)
print(cm)


