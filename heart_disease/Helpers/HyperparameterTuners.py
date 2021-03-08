from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier as KNN


def tune_KNN(df):
    features = df.drop(['label'], axis = 1)
    labels = df['label']
    X_train, X_test, y_train, y_test =  train_test_split(features, labels, test_size= 0.25, shuffle = True)


    k_range = list(range(3,20))
    weights = ["uniform", "distance"]
    algorithms = ['auto', 'ball_tree', 'kd_tree']
    leaf_sizes = [1,2,3,5,10,30] # played with these to find approx range
    param_grid = dict(n_neighbors = k_range, weights = weights, algorithm = algorithms, leaf_size = leaf_sizes)
    
    scores = ['roc_auc'] # I want to use roc/auc because I care more about getting all the positives

    for score in scores: # just using ROC_AUC for the comparison, but it's interesting to see different scores
        # this section is adapted from the example in the SKLearn docs
        print("# Tuning hyper-parameters for %s" % score)
        clf = GridSearchCV(
            KNN(), param_grid, cv = 5, scoring= score
        )
        clf.fit(X_train, y_train) 
        """
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        """
        print("Best KNN params: ", clf.best_params_)
        y_true, y_pred = y_test, clf.predict(X_test)
        #print(classification_report(y_true, y_pred))

        return clf.best_params_

def tune_SVC(df):
    features = df.drop(['label'], axis = 1)
    labels = df['label']
    X_train, X_test, y_train, y_test =  train_test_split(features, labels, test_size= 0.25, shuffle = True)

    tuning_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['roc_auc'] 

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
    
        clf = GridSearchCV(
            SVC(), tuning_params, cv = 5, scoring= score
        )
        clf.fit(X_train, y_train) 
        """
        print("Best parameters set found on development set:")
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        """
        print("Best SVC params: ", clf.best_params_)
        y_true, y_pred = y_test, clf.predict(X_test)
        #print(classification_report(y_true, y_pred))

        return clf.best_params_
