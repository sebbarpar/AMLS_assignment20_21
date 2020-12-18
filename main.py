import data_prep
import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
from data_prep import extract_feat_A, extract_feat_B
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
# ======================================================================================================================
# Data preprocessing
features, gender, smile = extract_feat_A()
#reshape features so they are not 2 dimensional
features1=features.reshape(features.shape[0],-1)

# ======================================================================================================================
# Task A1
xtrain, xtest, ytrain,ytest=train_test_split(features1, gender, random_state=0)
#First algorithm
logreg=LogisticRegression(solver='lbfgs', max_iter=10000, C=0.1)
#Second algorithm
ridge_regr_model = RidgeClassifier(alpha= 30,fit_intercept=True)
#Third algorithm
model=SVC(kernel='poly', C=1.5)
#Voting
eclf = VotingClassifier(estimators=[('svc', model),('rr',ridge_regr_model ), ('lr', logreg)], voting='hard',weights=[1,1,1])
eclf.fit(xtrain, ytrain)
#Accuracy
acc_A1_train=accuracy_score(ytrain, eclf.predict(xtrain))
acc_A1_test=accuracy_score(ytest, eclf.predict(xtest))           

# ======================================================================================================================
# Task A2
xtrain, xtest, ytrain,ytest=train_test_split(features1, smile, random_state=0)
#Reduce features
clf=RandomForestClassifier(n_estimators=68, max_depth=6, min_samples_split=6)
clf.fit(xtrain, ytrain)
imp=clf.feature_importances_
x=0
featuresA2=features1
for i in range(len(imp)):
    if (imp[i]<0.005):
        featuresA2=np.delete(featuresA2, i-x, 1)
        x=x+1
#Split dataset again
xtrain, xtest, ytrain,ytest=train_test_split(featuresA2, smile, random_state=0, train_size=0.35)
#Apply logistic regression
logreg_A1=LogisticRegression(solver='lbfgs', max_iter=10000, C=1.5)
logreg_A1.fit(xtrain, ytrain)
#Accuracy
acc_A2_train=accuracy_score(ytrain, logreg_A1.predict(xtrain))
acc_A2_test=accuracy_score(ytest, logreg_A1.predict(xtest))
#======================================================================================================================
#Free memory and extract second dataset
del features, gender, smile, features1, featuresA2, imp, x
img, eye, face=extract_feat_B() 
 # ======================================================================================================================
# Task B1
images=np.zeros((10000,3))
#Select point to evaluate RGB
x=270
y=200
for i in range(10000):
    images[i]=(img[i, x, y])
xtrain, xtest, ytrain, ytest=train_test_split(images, eye, random_state=0)
#Bagging
bagmodel=BaggingClassifier(n_estimators=30, max_samples=0.5, max_features=2)
bagmodel.fit(xtrain, ytrain)
#Accuracy
acc_train=accuracy_score(ytrain, bagmodel.predict(xtrain))
acc_test=accuracy_score(ytest, bagmodel.predict(xtest))

# ======================================================================================================================
# Task B2
#Canny edge algorithm and crop pictures
canny=[]
for i in range(len(img)):
    canny.append(cv2.Canny(img[i],100,200))
canny=np.array(canny)
canny2=np.zeros((10000,300,300),'uint8' )
for i in range(10000):
    canny2[i]=canny[i,100:400, 100:400]
canny1=canny2.reshape(10000,-1)
xtrain, xtest, ytrain,ytest=train_test_split(canny1, face, random_state=0, train_size=0.2)
logreg_B2=LogisticRegression(solver='lbfgs', tol=0.1)
logreg_B2.fit(xtrain, ytrain)
acc_train=accuracy_score(ytrain, logreg_B2.predict(xtrain))
acc_test=accuracy_score(ytest, logreg_B2.predict(xtest))
# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))