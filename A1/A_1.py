

def Task_A1(features, gender):
    #convert gender to 0-male 1-female
    gen1=np.zeros(len(gender))
    for i in range(len(gender)):
        gen1[i]=(int(gender[i])+1)/2
    #reshape features so they are not 2 dimensional
    features1=features.reshape(4795,-1)
    #split dataset into train, validation and test
    print("Split dataset")
    x1, xtest, y1,ytest=train_test_split(features1, gen1, random_state=0)
    xtrain, xvalidation, ytrain, yvalidation=train_test_split(x1, y1, random_state=0)
    #There will be 3 algorithms defined which will be combined by voting
    logreg=LogisticRegression(solver='lbfgs', max_iter=10000, C=0.1)
    logreg.fit(xtrain, ytrain)
    ridge_regr_model = RidgeClassifier(alpha= 30,fit_intercept=True)
    ridge_regr_model.fit(xtrain, ytrain)
    model=SVC(kernel='poly', C=1.5)
    model.fit(xtrain, ytrain)
    #Vote between all models with different weights
    eclf = VotingClassifier(estimators=[('svc', model),('rr',ridge_regr_model ), ('lr', logreg)], voting='hard',weights=[1,1,1])
    eclf.fit(xtrain, ytrain)
    y=eclf.predict(xtest)
    acc_train=accuracy_score(y1, eclf.predict(x1))
    acc_test=accuracy_score(ytest, eclf.predict(xtest))
    #uncomment next line to display learning curve
    print("Test accuracy score is:")
    print(accuracy_score(ytest, y))
    return eclf, acc_train, acc_test
    
def learning_curve_A1(eclf,x1,y1):
    #Learning curve
    train_sizes, train_scores, validation_scores=learning_curve(estimator=eclf, X=x1, y=y1, cv=5)
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)
    #Plot learning curve
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('Score')
    plt.xlabel('Training set size')
    plt.legend()