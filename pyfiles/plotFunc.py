#!/usr/bin/env python
# coding: utf-8


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import sklearn.metrics import confusion_matrix
from sklearn import tree


def barplot_general(targets, clf_names, title):
    '''
    title: time, scores...
    '''
    y_pos = np.arange(len(clf_names))
    plt.bar(y_pos, targets, color=(0.1, 0.1, 0.1, 0.1),  edgecolor='blue')
    plt.xticks(y_pos, clf_names)
    plt.title('Barplot for {} in different models'.format(title))
    plt.show

def barplot_stacked(train_err,test_err,names,N):
    # y-axis in bold
    rc('font', weight='bold')
    # Heights of bars1 + bars2
    bars = np.add(train_err, train_err).tolist()
    # The position of the bars on the x-axis
    r = np.array(range(N))
    barWidth = 1
    plt.bar(r, train_err, color='#7f6d5f', edgecolor='white', width=barWidth)
    plt.bar(r, test_err, bottom=train_err, color='#557f2d', edgecolor='white', width=barWidth)
    plt.xticks(r, names, fontweight='bold')
    plt.xlabel("Model")
    plt.ylabel('Errors')
    plt.legend(loc='upper right')
    plt.legend(labels=['Training', 'Testing'])
    plt.title('Errors by Model training and testing')
    plt.show()

def heat_map(dataset,path=True):
    if path:
        df = pd.read_csv(dataset)
        df = df.set_index('model')
    else:
        df=dataset
    sns.clustermap(df, metric="euclidean", standard_scale=1, method="ward", cmap="mako")

def roc_curve(fpr,tpr,scores, clf_name):
    ax = plt.subplot(111)
    plt.rcParams['figure.figsize'] = 10, 8
    for i in range(len(clf_name)):
        plt.plot(fpr[i],tpr[i],label = '{}, AUC = {}'.format(clf_name[i],scores[i]))
    plt.plot([0,1],[0,1],linestyle = '--',lw =1, color = 'blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('The ROC plot')
    plt.grid(True)
    plt.show()

def confus_matrix(test,pred):
    cm = confusion_matrix(test,pred)
    fig,ax = plt.subplots()
    for i in range(6):
        for j in range(6):
            c = cm[j,i]
            ax.text(i,j,str(c),va = 'center',ha = 'center')


def vis_tree(best_clf, x_train, y_train, max_step, feature_names,dt = True):
    if dt:
        best_clf.fit(x_train,y_train)
        dot_data = tree.export_graphviz(best_clf,max_step = max_step,
                                       filled = True,
                                       rounded = True,
                                       feature_names = list(feature_names),
                                       class_names = ['3','4','5','6','7','8','9','10'])
    else: #random forest
        best_clf.fit(x_train,y_train)
        estimator = best_clf.estimators_[5]
        dot_data = tree.export_graphviz(estimator,max_step = max_step,
                                       filled = True,
                                       rounded = True,
                                       feature_names = list(feature_names),
                                       class_names = ['3','4','5','6','7','8','9','10'])
    graph=graphviz.Source(dot_data)
    return(graph)