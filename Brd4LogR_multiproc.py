#--------------------------------------------------------------------------
# Name: Brd4LogR_multiproc.py
# Author: Yolanda
# Instruction: Build a logistic-regression classification model for Brd4
#     specific score-function and validate on testset. Plot the lgr-figure
#     of testset. Plot heatmap to compare testset predictability of Brd4_LGR
#     and Glide.
#---------------------------------------------------------------------------


import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from itertools import combinations
import multiprocessing as mul

DESCRIPTOR_all = ['140_np','140_polar','146_np','146_polar','81_np','81_polar','82_np','82_polar',\
              '83_np','83_polar','85_np','85_polar','87_np','87_polar','92_np','92_polar',\
              '94_np','94_polar','97_np','97_polar','140_hb','electro','w_hb']
train_file = ''
test_file = ''


def descriptorCom():
    for i in range(12,19):
        for des in combinations(DESCRIPTOR_all, i):
            yield list(des)


def loadData(datafile, descriptor=DESCRIPTOR_all):
    raw = [i.strip().split('\t') for i in open(datafile)]
    titles = raw[0]
    
    def entry(rawline): return dict(zip(titles, rawline))
    
    X = np.matrix([[float(entry(r)[i]) for i in descriptor] for r in raw[1:]])
    y = np.matrix([int(entry(r)['Active']) for r in raw[1:]]).T
    names = [entry(r)['Name'] for r in raw[1:]]
    return titles, X, y, names


def calRate(pred, testy):
    t = len([i for i in testy if int(i)==1])
    f = len([i for i in testy if int(i)==0])
    tp = len([(a,b) for a,b in zip(pred,testy) if int(a)==1 and int(b)==1])/float(t)
    tn = len([(a,b) for a,b in zip(pred,testy) if int(a)==0 and int(b)==0])/float(f)
    return tp, tn


def logRMod(trainX, trainy, testX=[], testy=[]):
    lr = LogisticRegression()
    lr.fit(trainX, trainy.flat)
    scores, N = [], len(trainy)
    kf = cross_validation.KFold(N, n_folds=10)
    for train_idx,test_idx in kf:
        X_train, X_test = trainX[train_idx], trainX[test_idx]
        y_train, y_test = trainy[train_idx], trainy[test_idx]
        lr1 = LogisticRegression()
        lr1.fit(X_train, y_train.flat)
        scores.append(lr1.score(X_test, y_test.flat))
    #print 'Train set cross validation scores:\n', sum(scores)/10.0
    
    if testX==[] or testy==[]: return lr
    prob = list(lr.predict_proba(testX))
    pred = list(lr.predict(testX))
    tp, tn = calRate(pred, list(testy.flat))
    #print 'Test set\nTrue positive:\n', tp
    #print 'True negtive:\n', tn
    score = lr.score(testX, testy.flat)
    auc = roc_auc_score(trainy.flat, [1-i[0] for i in lr.predict_proba(trainX)])
    auc_testset = roc_auc_score(testy.flat, [1-i[0] for i in lr.predict_proba(testX)])
    #print 'Right rate:\n', score
    return lr, auc, sum(scores)/10.0, score, tp, tn, auc_testset


def scoreGen(lr, data2pred):
    prob = [i[0] for i in list(lr.predict_proba(data2pred))]
    pred = list(lr.predict(data2pred))
    return prob, pred


def resPlot(lr, testX, testy):
    X_fit = list((testX.dot(lr.coef_.T)+lr.intercept_).flat)
    from math import exp
    e = exp(1)
    y_fit = [1-1/(1+e**x) for x in X_fit]
    xyy = sorted(zip(X_fit, y_fit, list(testy.flat)))
    xtp = [x for [x,y,y1] in xyy if int(y1)==1 and y>=0.5]
    ytp = [y for [x,y,y1] in xyy if int(y1)==1 and y>=0.5]
    xfp = [x for [x,y,y1] in xyy if int(y1)==0 and y>=0.5]
    yfp = [y for [x,y,y1] in xyy if int(y1)==0 and y>=0.5]
    xtn = [x for [x,y,y1] in xyy if int(y1)==0 and y<0.5]
    ytn = [y for [x,y,y1] in xyy if int(y1)==0 and y<0.5]
    xfn = [x for [x,y,y1] in xyy if int(y1)==1 and y<0.5]
    yfn = [y for [x,y,y1] in xyy if int(y1)==1 and y<0.5]
    Fig = plt.figure(figsize=(11,8))
    l = ['TruePositive','TrueNegative','FalsePositive','FalseNegative']
    plt.plot(xtp, ytp, 'o-g', alpha=0.8, lw=3, markersize=18, label=l[0])
    plt.plot(xtn, ytn, 'o-r', alpha=0.8, lw=3, markersize=18, label=l[1])
    plt.plot(xfp, yfp, '*r', markersize=18, label=l[2])
    plt.plot(xfn, yfn, '*g', markersize=18, label=l[3])
    plt.ylabel('Probability of Active (Predicted)', fontsize=20)
    plt.title('Predicted Probability on Test Data', fontsize=30)
    plt.legend(l, loc='lower right', fontsize=18)
    plt.plot([-6,8],[0.5,0.5],'-.k')
    #Fig.savefig('../dataSet/logrPlot_highres.png')
    plt.show()
    


def singleRun(descriptor):
    titles, X, y, names = loadData(train_file, descriptor=descriptor)
    titles, testX, testy, testNames = loadData(test_file, descriptor=descriptor)
    lr, auc, cv, score, tp, tn, auc_testset = logRMod(X, y, testX, testy)
    return [auc, cv, score, tp, tn, auc_testset, descriptor, list(lr.coef_.flat)]




t = mul.Pool(12)
res = []
best = []
f = open('desComRes_update160125.txt', 'w')
for n,i in enumerate(t.imap(singleRun, descriptorCom())):
    if n%10000==0: print n, '\t', len(res)
    if i[0]>0.8 and i[1]>0.7:
        res.append(i)
        best0 = max(res)
        if best < best0:
            best = best0
            print best
    if len(res)>5000:    
        for r in res[:5000]: f.write(str(r)+'\n')
        res = res[5000:]
for r in res: f.write(str(r)+'\n')
f.close()
DESCRIPTOR = best[6]

titles, X, y, names = loadData(train_file, descriptor=best[6])
titles, testX, testy, testNames = loadData(test_file, descriptor=best[6])
lr, auc, cv, score, tp, tn = logRMod(X, y, testX, testy)
print 'cv: %s\nscore: %s\ntp: %s\ntn: %s'%(cv,score,tp,tn)
for dsc,coef in zip(best[5], lr.coef_.flat): print dsc, coef
print 'Intercept:', lr.intercept_
#resPlot(lr, testX, testy)




