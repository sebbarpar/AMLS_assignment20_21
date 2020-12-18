
def B_1():
    img, eye, face=extract_feat_B()
    images=np.zeros((10000,3))
    x=270
    y=200
    for i in range(10000):
        images[i]=(img[i, x, y])
    x1, xtest, y1,ytest=train_test_split(images, eye, random_state=0)
    xtrain, xvalidation, ytrain, yvalidation=train_test_split(x1, y1, random_state=0)
    bagmodel=BaggingClassifier(n_estimators=30, max_samples=0.5, max_features=2)
    bagmodel.fit(xtrain, ytrain)
    print("Train accuracy score is:")
    print(accuracy_score(ytrain, bagmodel.predict(xtrain)))
    print("Validation accuracy score is:")
    print(accuracy_score(yvalidation, bagmodel.predict(xvalidation)))
    print("Test accuracy score is:")
    print(accuracy_score(ytest, bagmodel.predict(xtest)))
    #Uncomment next line to display learning curve
    #learning_curve_B1(bagmodel, x1, y1)
    acc_train=accuracy_score(y1, bagmodel.predict(x1))
    acc_test=accuracy_score(ytest, bagmodel.predict(xtest))
    return bagmodel, acc_train, acc_test 
    
def learning_curve_B1(bagmodel, x1, y1):
    train_sizes, train_scores, validation_scores=learning_curve(estimator=bagmodel, X=x1, y=y1, cv=5)
    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)
    plt.plot(train_sizes, -train_scores_mean, label = 'Training score')
    plt.plot(train_sizes, -validation_scores_mean, label = 'Cross-validation score')
    plt.ylabel('Score')
    plt.xlabel('Training set size')
    plt.legend()