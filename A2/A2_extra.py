#test which threshold to set when selecting features
xtrain, xtest, ytrain,ytest=train_test_split(features1, sm, random_state=0)
clf=RandomForestClassifier(n_estimators=k)
clf.fit(xtrain, ytrain)
y_predf=clf.predict(xtest)
print(accuracy_score(ytest, y_predf))
imp=clf.feature_importances_
#Possible threshold values
thresh=[0.0001, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
acc=np.zeros(len(thresh))
n_feat=np.zeros(len(thresh))
for j in range(len(thresh)):  
    features2=features1
    x=0
    for i in range(len(imp)):
        if (imp[i]<thresh[j]):
            #update features every time one is under the threshold
            features2=np.delete(features2, i-x, 1)
            x=x+1
    n_feat[j]=features2.shape[1]
    y=cross_val_predict(clf, features2, sm, cv=10)
    acc[j]=accuracy_score(sm, y)
#plot
fig, ax1 = plt.subplots()
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Feature number')
plt.plot(thresh, n_feat)
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:red')
plt.plot(thresh, acc, color='tab:red') 
