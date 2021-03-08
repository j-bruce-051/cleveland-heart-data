from sklearn import model_selection, neighbors, svm
from sklearn.metrics import recall_score, f1_score


def use_KNN(df, dict):
    features = df.drop(['label'], axis = 1)
    labels = df['label']

    algorithm, leaf_size, n_neighbors, weights = dict['algorithm'], dict['leaf_size'], dict['n_neighbors'], dict['weights']
    x_train, x_test, y_train, y_test =  model_selection.train_test_split(features, labels, test_size= 0.25, shuffle = True)
    classifier = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, algorithm=algorithm, leaf_size=leaf_size, weights=weights)
    classifier = classifier.fit(x_train, y_train)
    acc = classifier.score(x_test, y_test)
    y_predict = classifier.predict(x_test)
    recall = recall_score(y_test, y_predict)

    """
    # sometimes this is nice to know but not all the time
    print("recall: ", recall)
    f1 = f1_score(y_test, y_predict)
    print('f1: ', f1)
    print( "accuracy: ", acc)
    """

    return acc, recall

def use_SVC(df, dict):

    kernel, C = dict['kernel'], dict['C']

    features = df.drop(['label'], axis = 1)
    labels = df['label']
    x_train, x_test, y_train, y_test =  model_selection.train_test_split(features, labels, test_size= 0.25, shuffle = True)
    if kernel == 'rbf':
        classifier = svm.SVC(kernel= kernel, C=C, gamma= dict['gamma'])
    else: 
        classifier = svm.SVC(kernel= kernel, C=C)
    classifier = classifier.fit(x_train, y_train)


    y_predict = classifier.predict(x_test)
    recall = recall_score(y_test, y_predict)
    acc = classifier.score(x_test, y_test)
    """
    print("recall: ", recall)
    f1 = f1_score(y_test, y_predict)
    print('f1: ', f1)
    print( "accuracy: ", acc)
    """
    return acc, recall