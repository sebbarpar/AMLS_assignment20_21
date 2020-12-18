
def B_2(img, face):
    #apply canny edge detection to decrease variables
    canny=[]
    for i in range(len(img)):
        canny.append(cv2.Canny(img[i],100,200))
    canny=np.array(canny)
    canny2=np.zeros((10000,300,300),'uint8' )
    #crop pictures
    for i in range(10000):
        canny2[i]=canny[i,100:400, 100:400]
    canny1=canny2.reshape(10000,-1)
    print(canny1.shape)
    x1, xtest, y1,ytest=train_test_split(canny1, face, random_state=0, train_size=0.8, test_size=0.2)
    xtrain, xvalidation, ytrain, yvalidation=train_test_split(x1, y1, random_state=0, train_size=0.25, test_size=0.75)
    #train logistic regression algorithm
    logreg=LogisticRegression(solver='lbfgs', tol=0.1)
    logreg.fit(xtrain, ytrain)
    #learning_curve_B2(logreg, x1, y1)
    print("Train accuracy score is:")
    print(accuracy_score(ytrain, logreg.predict(xtrain)))
    print("Validation accuracy score is:")
    print(accuracy_score(yvalidation, logreg.predict(xvalidation)))
    print("Test accuracy score is:")
    print(accuracy_score(ytest, logreg.predict(xtest)))
    
def learning_curve_B2(estim, x1, y1):
    train_sizes, train_scores, validation_scores=learning_curve(estimator=estim, X=x1, y=y1, cv=5)
    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)
    plt.plot(train_sizes, -train_scores_mean, label = 'Training score')
    plt.plot(train_sizes, -validation_scores_mean, label = 'Cross-validation score')
    plt.ylabel('Score')
    plt.xlabel('Training set size')
    plt.legend()