def validation():
    #Possible parameter values
    c=([0.1, 0.2, 0.3, 0.5, 0.8, 1])
    acc_t=np.zeros(len(c))
    acc=np.zeros(len(c))
    for i in range(len(c)):
        model=LogisticRegression(solver='lbfgs', max_iter=10000, C=c[i])
        model.fit(xtrain, ytrain)
        y_v=model.predict(xvalidation)
        acc[i]=accuracy_score(yvalidation, y_v)
        acc_t[i]=accuracy_score(ytrain, model.predict(xtrain))
    #plot values against accuracy
    fig, ax1 = plt.subplots()
    plt.plot(c, acc)
    ax1.set_xlabel('C')
    ax1.set_ylabel('Accuracy')
    plt.plot(c, acc_t)
    fig.legend(["Validation", "Training"])