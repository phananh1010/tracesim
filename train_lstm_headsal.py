import headpred
import numpy as np
import pickle
import copy
from keras.callbacks import ModelCheckpoint
import tracesim_header


import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import pickle

from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

def create_vector_dataset(sds, step_back=8, step_ahead=8):
    #for lingress, only differences is data are flatten
    
    datX = []
    datY = []
    model = '360net'
    #R = 2
    for topic in ['1', '2', '3', '4', '5', '6']:#['1', '2']:#
        timelist = sds[model][topic]['timestamp']
        sallist = sds[model][topic]['salient']
        headposdict = sds[model][topic]['headpos']
        for uid in range(40):
            for idx in range(len(sallist)):

                headpos = headposdict[uid][idx]
                sal = np.copy(sallist[idx]) + np.random.random(size=sallist[idx].shape)*0.0001
                sal = sal/sal.max()
                datX.append([headpos.ravel().tolist() + sal.ravel().tolist()])
                datY.append(headpos)
    datX = np.array(datX)
    datY = np.array(datY)
    print datX[0].shape
    
    L = len(datX)
    step = step_back + step_ahead
    X = []
    y = []
    for i in range(L - step):
        X.append(datX[i:i+step_back])
        y.append(datY[i:i+step][-1])
        
    X = np.array(X)#N, C, H, W. C is sample in the past (step_back)
    X = X.reshape(X.shape[0], X.shape[1], -1)
    
    y = np.array(y)
    y = y.reshape(y.shape[0], -1)
    return X, y

def gen_vector_dataset(sds, _R, _step_back, _step_ahead, _model='lstm', _idx_list=[]):
    #this include create datX, datY from sds datastructure
    #this simply convert salist (2d map) and head position (vector) into data structure for training

    
    
    X0, y0 = create_vector_dataset(sds, _step_back, _step_ahead)

    
    #shuffle the created datastructure
    if _idx_list == []:
        idx_list = range(len(X0))
        np.random.shuffle(idx_list)
    else:
        print 'Using external idx_list'
        idx_list = _idx_list
    
    X0 = X0[idx_list]
    y0 = y0[idx_list]
    
    #split 
    split_pos = int(len(X0)*(.9))
    Xtrain = X0[:split_pos]
    ytrain = y0[:split_pos]
    Xtest = X0[split_pos:]
    ytest = y0[split_pos:]  
    
    return Xtrain, ytrain, Xtest, ytest


def rebalance_vector_trainingset(_X, _y):
    #remove non moving data from training set, to make model smarter
    X0, y0, X1, y1, X2, y2, X3, y3, X4, y4, X5, y5 = [[] for i in range(12)]
    for idx, _ in enumerate(_X):
        xpos = _X[idx][-1][:3]
        ypos = _y[idx]
        if headpred.degree_distance(xpos, ypos) > 60:
            X1.append(_X[idx])
            y1.append(_y[idx])
        elif headpred.degree_distance(xpos, ypos) > 30:
            X2.append(_X[idx])
            y2.append(_y[idx])
        elif headpred.degree_distance(xpos, ypos) > 10:
            X3.append(_X[idx])
            y3.append(_y[idx])
        elif headpred.degree_distance(xpos, ypos) > 5:
            X4.append(_X[idx])
            y4.append(_y[idx])
        else:
            X5.append(_X[idx])
            y5.append(_y[idx])
    X0 = copy.deepcopy(X1)
    y0 = copy.deepcopy(y1)
    X0 = np.vstack((X0, X2[:len(X1)]))
    y0 = np.vstack((y0, y2[:len(y1)]))
    X0 = np.vstack((X0, X3[:len(X1)]))
    y0 = np.vstack((y0, y3[:len(y1)]))
    X0 = np.vstack((X0, X4[:len(X1)/2]))
    y0 = np.vstack((y0, y4[:len(y1)/2]))
    X0 = np.vstack((X0, X5[:len(X1)/8]))
    y0 = np.vstack((y0, y5[:len(y1)/8]))
    
    tidxl = np.arange(len(y0))
    np.random.shuffle(tidxl)
    X0 = X0[tidxl]
    y0 = y0[tidxl]
    
    #return np.array(X1)[:len(y1)/2], np.array(y1[:len(y1)/2])
    return np.array(X0), np.array(y0)


def eval_acc(_fx, _y, _visualize=False):
    h=9
    w=16
    R=2
    
    result_acc = []
    for idx, _ in enumerate(_fx):
        fxi = headpred.create_fixblur_map(_fx[idx], h, w, R)
        yi = headpred.create_fixblur_map(_y[idx], h, w, R)
        
        if _visualize==True:
            plt.imshow(fxi.reshape(9, 16))
            plt.figure()
            plt.imshow(yi.reshape(9, 16))
            plt.figure()
        
        result_acc.append(headpred.overlap_accuracy(fxi, yi, _visualize=_visualize))
    return result_acc

def eval_mapbaseline(_map_list, _y, _visualize=False):
    h=9
    w=16
    R=2
    
    result_acc = []
    for idx, _ in enumerate(_map_list):
        yi = headpred.create_fixblur_map(_y[idx], h, w, R)
        
        if _visualize==True:
            plt.imshow(_map_list[idx])
            plt.figure()
            plt.imshow(yi.reshape(9, 16))
            plt.figure()
        
        result_acc.append(headpred.overlap_accuracy(_map_list[idx], yi, _visualize=_visualize))
    return result_acc    

def get_lnregr_model():
    sds = pickle.load(open(tracesim_header.FILE_PATH_SDS))

    model = '360net'
    alien = '2'
    skiing = '1'
    sds[model][alien].keys()
    uid = 17

    time_list = sds[model][alien]['timestamp']
    sal_list = sds[model][alien]['salient']
    headpos_list = sds[model][alien]['headpos'][uid]
    headpos_dict = sds[model][alien]['headpos']
    file_list = sds[model][alien]['salient_filename']
    uid = 17


    look_back = 8
    look_ahead = 8

    idxlist = np.arange(10 * (len(sal_list) - look_back - look_ahead))
    np.random.shuffle(idxlist)

    R = 2
    Xtrain, ytrain, Xtest, ytest = gen_vector_dataset(sds, R, look_back, look_ahead, _model='linear')

    Xtrain, ytrain = rebalance_vector_trainingset(Xtrain, ytrain)
    Xtrain = Xtrain[:20000]
    ytrain = ytrain[:20000]



    lnr = LinearRegression()
    knn = KNeighborsRegressor()
    regr_lnr = headpred.multi_train(Xtrain.reshape(Xtrain.shape[0], -1), ytrain, model=lnr)
    #fx_lnr = regr_lnr.predict(Xtest.reshape(Xtest.shape[0], -1))
    return regr_lnr

