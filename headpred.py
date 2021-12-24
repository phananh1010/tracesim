import numpy as np
import pickle
import hp_header
import cv2

from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def degree_distance(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi * 180

def vector_to_ang(_v):
    #v = np.array(vector_ds[0][600][1])
    #v = np.array([0, 0, 1])
    _v = np.array(_v)
    alpha = degree_distance(_v, [0, 1, 0])#degree between v and [0, 1, 0]
    phi = 90.0 - alpha
    proj1 = [0, np.cos(alpha/180.0 * np.pi), 0] #proj1 is the projection of v onto [0, 1, 0] axis
    proj2 = _v - proj1#proj2 is the projection of v onto the plane([1, 0, 0], [0, 0, 1])
    theta = degree_distance(proj2, [1, 0, 0])#theta = degree between project vector to plane and [1, 0, 0]
    sign = -1.0 if degree_distance(_v, [0, 0, -1]) > 90 else 1.0
    theta = sign * theta
    return theta, phi

def ang_to_geoxy(_theta, _phi, _h, _w):
    x = _h/2.0 - (_h/2.0) * np.sin(_phi/180.0 * np.pi)
    temp = _theta
    if temp < 0: temp = 180 + temp + 180
    temp = 360 - temp
    y = (temp * 1.0/360 * _w)
    return int(x), int(y)
    

def create_fixation_map(_headpos, _H, _W):
    theta, phi  = vector_to_ang(_headpos)
    theta ,phi = -theta, -phi
    hi, wi = ang_to_geoxy(theta, phi, _H, _W)
    result = np.zeros(shape=(_H, _W))
    result[hi, wi] = 1
    return result

def create_fixation_map2(_headpos, _H, _W):
    theta, phi  = vector_to_ang(_headpos)
    theta ,phi = -theta, -phi
    hi, wi = ang_to_geoxy(theta, phi, _H, _W)
    result = np.zeros(shape=(_H, _W))
    
    result[_H/2, _W/2] = 1
    result = np.roll(result, hi - _H/2, 0)
    result = np.roll(result, wi - _W/2, 1)
    return result

def create_fixblur_map(_headpos, _H, _W, _R):
    theta, phi  = vector_to_ang(_headpos)
    theta ,phi = -theta, -phi
    hi, wi = ang_to_geoxy(theta, phi, _H, _W)
    result = np.zeros(shape=(_H, _W))
    
    result[_H/2, _W/2] = 1
    result = cv2.GaussianBlur(result, (2*_R + 1, 2*_R + 1), 0)
    result = np.roll(result, hi - _H/2, 0)
    result = np.roll(result, wi - _W/2, 1)
    return result

########fixation map to fov###########
def get_range(_i, _r, _n):
    pos_s = _i - _r
    pos_e = _i + _r
    #if pos_s < 0: 
    #    pos_s = _n + pos_s
    #if pos_e >= _n:
    #    pos_e = pos_e - _n
        
    result = [item%_n for item in range(pos_s, pos_e+1)]
    return result

def roll_w(_headmap, _pos_w, _pos_h):
    result = np.roll(_headmap, _pos_w, 1)
    result = np.roll(result, _pos_h, 0)
    return result

def expand(_headmap, _r):
    _h, _w = _headmap.shape
    _hi, _wi = np.unravel_index(_headmap.argmax(), _headmap.shape)
    
    result = np.zeros(shape=(_h, _w))
    col_list = get_range(_wi, _r, _w)
    for col in col_list:
        row_list = get_range(_hi, _r, _h)
        for row in row_list:
            result[row][col] = 1
    return result

#def create_dataset(_data, step_back=8, step_ahead=8):
#    L = len(_data)
#    step = step_back + step_ahead
#    X = []
#    y = []
#    for i in range(L - step):
#        X.append(_data[i:i+step_back])
#        y.append(_data[i:i+step][-1])
#        
#    X = np.array(X)
#    X = X.reshape(X.shape[0], -1)
#    
#    y = np.array(y)
#    y = y.reshape(y.shape[0], -1)
#    return X, y

def create_dataset_lingress(_datX, _datY, step_back=8, step_ahead=8):
    #for lingress, only differences is data are flatten
    L = len(_datX)
    step = step_back + step_ahead
    X = []
    y = []
    for i in range(L - step):
        X.append(_datX[i:i+step_back])
        y.append(_datY[i:i+step][-1])
        
    X = np.array(X)
    X = X.reshape(X.shape[0], -1)
    
    y = np.array(y)
    y = y.reshape(y.shape[0], -1)
    return X, y

def create_dataset_lstm(_datX, _datY, step_back=8, step_ahead=8):
    #for lingress, only differences is data are flatten
    L = len(_datX)
    step = step_back + step_ahead
    X = []
    y = []
    for i in range(L - step):
        X.append(_datX[i:i+step_back])
        y.append(_datY[i:i+step][-1])
        
    X = np.array(X)#N, C, H, W. C is sample in the past (step_back)
    X = X.reshape(X.shape[0], X.shape[1], -1)
    
    y = np.array(y)
    y = y.reshape(y.shape[0], -1)
    return X, y

def multi_train(_X, _y, model=SVR()):
    regr = MultiOutputRegressor(model)

    regr.fit(_X, _y)    
    return regr

def gen_dataset(_sal_list, _headpos_dict, _R, _step_back, _step_ahead, _model='lstm', _idx_list=[]):
    #this include create datX, datY from sds datastructure
    #this simply convert salist (2d map) and head position (vector) into data structure for training
    datX = []
    datY = []
    #R = 2
    for uid in range(10):
        for idx in range(len(_sal_list)):

            headpos = _headpos_dict[uid][idx]
            sal = sal_list[idx]

            h, w = sal_list[idx].shape
            
            #fixmap = create_fixation_map(headpos, h, w)
            fixmap2 = create_fixation_map2(headpos, h, w)
            fixblur = create_fixblur_map(headpos, h, w)
            headmap = expand(fixmap2, R)
            #salmap = expand(sal, R)

            #temp = np.vstack((fixblur, sal))
            datX.append([sal, fixblur])
            datY.append(fixblur)
            


    
    #from datX, datY, create the dataset
    if _model == 'lstm':
        X0, y0 = create_dataset_lstm(datX, datY, _step_back, _step_ahead)
    elif _model == 'linear':
        X0, y0 = create_dataset_lingress(datX, datY, _step_back, _step_ahead)
    else:
        X0, y0 = None, None
    
    #shuffle the created datastructure
    if _idx_list == []:
        idx_list = range(len(X0))
        np.random.shuffle(idx_list)
    else:
        print ('Using external idx_list')
        idx_list = _idx_list
    
    X0 = X0[idx_list]
    y0 = y0[idx_list]
    
    #split 
    split_pos = int(len(X0)*(.666))
    Xtrain = X0[:split_pos]
    ytrain = y0[:split_pos]
    Xtest = X0[split_pos:]
    ytest = y0[split_pos:]  
    
    return Xtrain, ytrain, Xtest, ytest

def visualize_predict(_Xtest, _ytest, _fx, _pos):
    xtest = _Xtest[_pos].reshape(8, 2, 9, 16)[-1]
    

    gs1 = gridspec.GridSpec(2, 4)
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    for i, xtest in enumerate(_Xtest[_pos].reshape(8, 2, 9, 16)):
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        plt.subplot(gs1[i])
        plt.axis('off')
        plt.imshow(xtest[0])
        
    plt.figure()  
    for i, xtest in enumerate(_Xtest[_pos].reshape(8, 2, 9, 16)):
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        plt.subplot(gs1[i])
        plt.axis('off')
        plt.imshow(xtest[1])
    
    #special puropse code
    t = _Xtest[_pos].reshape(8, 2, 9, 16)[-1]
    plt.figure()
    
    plt.axis('off')
    plt.imshow(t[0])
    plt.savefig('lstm_input_saliency.png')
    plt.figure()
    
    plt.axis('off')
    plt.imshow(t[1])
    plt.savefig('lstm_input_head_orientation.png')
    plt.figure()
    
    plt.imshow(_fx[_pos].reshape(9, 16))
    plt.axis('off')
    plt.savefig('lstm_output_prediction.png')
    plt.figure()
    
    plt.imshow(expand(_fx[_pos].reshape(9, 16), 2))
    plt.axis('off')
    plt.savefig('lstm_output_viewport.png')
    plt.figure()
    
    plt.imshow(_ytest[_pos].reshape(9, 16))
    plt.axis('off')
    plt.savefig('grouthtruth.png')
    plt.figure()
    print ('input saliency', 'input head orientation map', 'model prediction', 'ground truth')

def count_black(t1, t2):
    intersect = np.array(np.logical_and(t1, t2), dtype=int)
    union = np.array(np.logical_or(t1, t2), dtype=int)
    
    return intersect.sum() * 1.0 / union.sum()  
    
def overlap_ratio(t1, t2):
    intersect = np.array(np.logical_and(t1, t2), dtype=int)
    union = np.array(np.logical_or(t1, t2), dtype=int)
    
    if union.sum() == 0:
        return 0.0
    else:
        return intersect.sum() * 1.0 / union.sum()    

def viewport_overlap(t1, t2):
    #t1 must be viewport
    intersect = np.array(np.logical_and(t1, t2), dtype=int)
    
    if intersect.sum() == 0:
        return 0.0
    else:
        return intersect.sum() * 1.0 / t1.sum()
    
    
def intersect_tilemap(vp, pd):
    #return the data showed inside viewport from prediction
    intersect = np.array(np.logical_and(vp, pd), dtype=int)
    
    return intersect

    
def overlap_accuracy(_map1, _map2, _R=2, _visualize=False):
    t1 = expand(_map1.reshape(9, 16), _R)
    t2 = expand(_map2.reshape(9, 16), _R)
    intersect = np.array(np.logical_and(t1, t2), dtype=int)
    union = np.array(np.logical_or(t1, t2), dtype=int)
    
    if _visualize == True:
        
        from matplotlib import pyplot as plt
        plt.imshow(intersect)
        plt.figure()
        plt.imshow(union)
        plt.figure()
    
    return intersect.sum() * 1.0 / union.sum()


def d_manha(map1, map2):
    h1, w1 = np.unravel_index(map1.argmax(), map1.shape)
    h2, w2 = np.unravel_index(map2.argmax(), map2.shape)
    
    return np.abs(h1 - h2) + np.abs(w1 - w2)

