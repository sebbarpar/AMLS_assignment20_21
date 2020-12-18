
def A_2(features, smile):
    #convert gender to 0-male 1-female
    sm=np.zeros(len(smile))
    for i in range(len(smile)):
        sm[i]=(int(smile[i])+1)/2
    #reshape features so they are not 2 dimensional
    features1=features.reshape(4795,-1)
    #split dataset into test and train
    x1, xtest, y1,ytest=train_test_split(features1, sm, random_state=0)
    xtrain, xvalidation, ytrain,yvalidation=train_test_split(x1, y1, random_state=0)
    clf=RandomForestClassifier(n_estimators=68, max_depth=6, min_samples_split=6)
    clf.fit(xtrain, ytrain)
    imp=clf.feature_importances_
    x=0
    features2=features1
    for i in range(len(imp)):
        if (imp[i]<0.005):
            features2=np.delete(features2, i-x, 1)
            x=x+1
    x1, xtest, y1,ytest=train_test_split(features2, sm, random_state=0, train_size=0.8)
    xtrain, xvalidation, ytrain,yvalidation=train_test_split(x1, y1, random_state=0, train_size=0.4)
    logreg=LogisticRegression(solver='lbfgs', max_iter=10000, C=1.5)
    logreg.fit(xtrain, ytrain)
    acc_train=accuracy_score(y1, logreg.predict(x1))
    acc_test=accuracy_score(ytest, logreg.predict(xtest))
    print("Train accuracy score is:")
    print(accuracy_score(ytrain, logreg.predict(xtrain)))
    print("Validation accuracy score is:")
    print(accuracy_score(yvalidation, logreg.predict(xvalidation)))
    print("Test accuracy score is:")
    print(accuracy_score(ytest, logreg.predict(xtest)))
    #Uncomment next line to display learning curve
    #learning_curve_A2(clf, x1, y1)
    return clf, acc_train, acc_test
    
def learning_curve_A2(clf, x1,y1):
    train_sizes, train_scores, validation_scores=learning_curve(estimator=clf, X=x1, y=y1, cv=5)
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)
    #Plot learning curve
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('Score')
    plt.xlabel('Training set size')
    plt.legend()    
   
