import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, classification_report

def main():
    # read data
    df = pd.read_csv('data.csv')
    features = df.columns # save features for future use
    x = df.to_numpy()
    print(x.shape) # check data dimension
    y = pd.read_csv('y.csv').to_numpy()
    print('read')
    
    # decision tree model
    # hyperparameter tuning for dt
    tuning(x, y)
    # performance results for dt with optimal parameters
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=25, min_samples_leaf=25)
    assessment(dt, x, y)
    metric_report(dt, x, y)
    feature_importance(dt, x, y, features)

    # naive bayes model
    gaus = GaussianNB()
    assessment(gaus, x, y)
    metric_report(gaus, x, y)

# print and plot feature importance for decision tree
def feature_importance(dt, x, y, features):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
    dt.fit(x_train, y_train)
    # use dataframe to store importances
    df = pd.DataFrame(data={'importance': dt.feature_importances_}, index=features)
    df = df.sort_values(by='importance', ascending=False) # sort importance in descending order
    
    # top 20
    top = df.head(20) # get top 20 most important features
    print(top.to_string)
    # plot top 20
    ax = plt.subplots()
    x = range(20)
    features = top.index
    importance = top['importance']
    ax.barh(x, importance, tick_label=features)
    ax.invert_yaxis()
    ax.set_xlabel('importance')
    plt.show()

    # bottom 20
    bot = df.tail(20)
    print(bot.to_string)
    # plot bottom 20
    ax = plt.subplots()
    x = range(20)
    features = bot.index
    importance = bot['importance']
    ax.barh(x, importance, tick_label=features)
    ax.invert_yaxis()
    ax.set_xlabel('importance')
    plt.show()



# print classification report on 80-20 split
def metric_report(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, labels=[1,2,3,4]))

# plot each metric for a changing parameter, xlabel
def plot_metrics(x, xlabel, acc, avg_pr, f1): 
    plt.plot(x, acc, label='acc')
    plt.plot(x, avg_pr, label='auprc')
    plt.plot(x, f1, label='f1')
    plt.xlabel(xlabel)
    plt.ylabel('metrics')
    plt.legend()
    plt.show()

# perform 10 fold assessment on model
def assessment(model, x, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=10)

    results = {}
    # record sum for each metric
    acc = 0
    avg_pr = 0
    f1 = 0
    # 10 fold cv
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        # calculate accuracy, auprc and f1 for fold
        results[i] = {'acc': accuracy_score(y_test, y_pred),
                      'average_precision': average_precision_score(y_test, model.predict_proba(x_test), average='macro'),
                      'f1': f1_score(y_test, y_pred, average='macro', labels=[1,2,3,4], zero_division=0)}
        acc += results[i]['acc']
        avg_pr += results[i]['average_precision']
        f1 += results[i]['f1']
    # print average for each metric
    print(results)
    print('acc')
    print(acc/10.0)
    print('avg_pr')
    print(avg_pr/10.0)
    print('f1')
    print(f1/10.0)
    # return average metrics
    return {'acc': acc/10.0, 'avg_pr': avg_pr/10.0, 'f1': f1/10.0}

# hyperparameter tuning with gridsearch --not used 
def gridsearch(model, x, y, parameters, scoring):
    grid = GridSearchCV(model, parameters, scoring=scoring, n_jobs=-1)
    grid.fit(x, y)
    print(grid.best_score_)
    print(grid.best_params_)

# hyperparameter tuning by parameter
def tuning(x, y):
    # criterion
    gini = DecisionTreeClassifier(criterion='gini', max_depth=100, min_samples_leaf=1)
    entropy = DecisionTreeClassifier(criterion='entropy', max_depth=100, min_samples_leaf=1)
    print('gini')
    assessment(gini, x, y)
    print('entropy')
    assessment(entropy, x, y)
    # max depth
    depth = [1, 5, 10, 20, 25, 30, 40, 50, 75, 100]
    acc = []
    avg_pr = []
    f1 = []
    for d in depth:
        print('depth', d)
        dt = DecisionTreeClassifier(criterion='entropy', max_depth=d, min_samples_leaf=1)
        results = assessment(dt, x, y)
        acc.append(results['acc'])
        avg_pr.append(results['avg_pr'])
        f1.append(results['f1'])
    plot_metrics(depth,'max_depth', acc, avg_pr, f1)
    # min_samples_leaf
    leaf = [1, 5, 10, 15, 20, 25, 30, 35]
    acc = []
    avg_pr = []
    f1 = []
    for l in leaf:
        print('leaf ', l)
        dt = DecisionTreeClassifier(criterion='entropy', max_depth=25, min_samples_leaf=l)
        results = assessment(dt, x, y)
        acc.append(results['acc'])
        avg_pr.append(results['avg_pr'])
        f1.append(results['f1'])
    plot_metrics(leaf, 'min_samples_leaf', acc, avg_pr, f1)

if __name__ == "__main__":
    main()