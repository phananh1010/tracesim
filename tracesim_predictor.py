import pickle
import numpy as np
import tracesim_header
import headpred
import train_lstm_headsal



class Predictor:
    _topic_dict = {'1': 'skiing.mp4', '0': 'conan1.mp4', '3': 'conan2.mp4', '2': 'alien.mp4','4': 'surfing.mp4', '7': 'football.mp4', '6': 'cooking.mp4', '8': 'rhinos.mp4'} #'5': 'war.mp4',
    _tilesize_map_dict = {}
    _tilemap_filetemplate = ''#'./dat/tilesize_map_{}'
    _sds_filepath = ''#'../testing/salient_ds_dict_w16_h9'
    _lstm_filetemplate = ''#'./models/lstm_128128_lookahead{}'
    _look_back = -1
    _look_ahead = -1
    _sds = {}
    _dataset = {} #dataset is a dict of datX, datY for individual video
    _lstm = None
    _lnregr = None
    
    LSTM = 'lstm'
    REGR = 'lnregr'
    SAL = 'sal'
    
    def __init__(self, tilemap_filetemplate, sds_filepath, lstm_filetemplate, look_back, look_ahead):
        self._tilemap_filetemplate = tilemap_filetemplate
        self._sds_filepath = sds_filepath
        self._lstm_filetemplate = lstm_filetemplate
        
        self._look_back = look_back
        self._look_ahead = look_ahead
        
        self._tilesize_map_dict = self.load_tilesize_map_dict()
        self._sds = self.load_sds()
        
        self._lstm = self.load_lstm(look_back, look_ahead)
        self._lnregr = train_lstm_headsal.get_lnregr_model()
        
    #########DATA PREPARATION###########
        
    def load_tilesize_map(self, tag):
        file_path_template = self._tilemap_filetemplate
        tilesize_map_filename = file_path_template.format(tag)
        return pickle.load(open(self._tilemap_filetemplate.format(tag), 'rb'))


    def load_tilesize_map_dict(self):
        H_list = [0,  160,  320,  480,  640,  800,  960, 1120, 1280]
        W_list = [0,  160,  320,  480,  640,  800,  960, 1120, 1280, 1440, 1600,
               1760, 1920, 2080, 2240, 2400]

        tag_dict = {k:self._topic_dict[k].replace('.mp4', '') for k in self._topic_dict}
        self._tilesize_map_dict = {tag: self.load_tilesize_map(tag_dict[tag]) for tag in tag_dict}
        return self._tilesize_map_dict
    
    def load_sds(self):
        self._sds = pickle.load(open(self._sds_filepath))['360net']
        return self._sds
    
    
    def get_start_pos(self, _time_list):
        #preprocessing, private function, to get starting position inside sds
        cut_pos = -1
        for idx, _ in enumerate(_time_list[:-1]):
            if _time_list[idx] > _time_list[idx+1]:
                cut_pos = idx + 1
                break
        if cut_pos == -1:
            cut_pos = 0
        return cut_pos
    
    def match_time(self, topic, t_time):
        #given any time, find the best next position in dataset, 
        #sothat predictor could produce a feature for the model
        timelist = self._sds[topic]['timestamp']
        start_pos = self.get_start_pos(timelist)

        return [[start_pos + idx, item] for idx, item in enumerate(timelist[start_pos:]) if item >= t_time][0]

    def create_sample(self, topic, uid, t_time, step_back, step_ahead):
        #given time pos, for example
        timelist = self._sds[topic]['timestamp']
        sallist = self._sds[topic]['salient']
        headposdict = self._sds[topic]['headpos']
       
        time_pos, p_time = self.match_time(topic, t_time)
        
        xi = []
        
        for i in range(time_pos - step_back - step_ahead, time_pos - step_ahead):
            headpos = headposdict[uid][i]
            sal = np.copy(sallist[i]) + np.random.random(size=sallist[i].shape)*0.0001
            sal = sal/sal.max()

            xi.append([headpos.ravel().tolist() + sal.ravel().tolist()])
        
        yi = headposdict[uid][time_pos]

        return np.array(xi).reshape(-1, 147), np.array(yi)
    
    def create_sample_v0(self, topic, uid, t_time, step_back, step_ahead):
        #given time pos, for example
        timelist = self._sds[topic]['timestamp']
        sallist = self._sds[topic]['salient']
        headposdict = self._sds[topic]['headpos']
       
        time_pos, p_time = self.match_time(topic, t_time)
        
        xi = []
        
        for i in range(time_pos - step_back - step_ahead, time_pos - step_ahead):
            headpos = headposdict[uid][i]
            sal = np.copy(sallist[i]) + np.random.random(size=sallist[i].shape)*0.0001
            sal = sal/sal.max()

            xi.append([headpos.ravel().tolist() + sal.ravel().tolist()])
        
        yi = headposdict[uid][time_pos]

        return np.array(xi).reshape(-1, 147), np.array(yi)
        
    def load_lstm(self, step_back, step_ahead):
        #load all 8sec, 16sec, 24sec and 32 secs model
        #then based on the step_ahead, use the selected model to 
        from keras import backend as K
        from keras import models, layers, optimizers, regularizers
        K.set_image_dim_ordering('th')
 
        H, W = tracesim_header.H, tracesim_header.W
        n_neurons = 128
        RW = 0.0001

        model3 = models.Sequential()
        model3.add(layers.LSTM(n_neurons, input_shape=(step_back, 3 + H * W), return_sequences=True, kernel_regularizer=regularizers.l2(RW)))
        model3.add(layers.LSTM(n_neurons, kernel_regularizer=regularizers.l2(RW)))
        model3.add(layers.Dense(3))
        rms = optimizers.RMSprop(lr=0.001)
        model3.compile(loss='mean_squared_error', optimizer=rms)#'adam')
        
        print 'LOADING {}'.format(self._lstm_filetemplate.format(step_ahead))
        model3.load_weights(self._lstm_filetemplate.format(step_ahead))
        
        return model3
 
    ############MAIN FUNCTIONALITY###########
    
    def get_VTIME_0(self, video_name):
        #return the begning time of specific videos for the framework
        series = self._sds[video_name]['timestamp']
        result = self.get_start_pos(series) + self._look_back + self._look_ahead
        
        return int(series[result] + 0.5)
    
    def get_VTIME_N(self, video_name):
        series = self._sds[video_name]['timestamp']
        result = self.get_start_pos(series[-20:])#20 index means around 1.3 seconds
        
        return int(series[-20:][result])
    
    def get_current_headpos(self, topic, uid, vtime):
        timelist = self._sds[topic]['timestamp']
        sallist = self._sds[topic]['salient']
        headposdict = self._sds[topic]['headpos']
        
        t_pos, t_time = self.match_time(topic, vtime)
        headpos = headposdict[uid][t_pos]
        return headpos
        
    def get_current_tilemap(self, topic, uid, v_time, _radius=tracesim_header.TILE_EXPAND_RADIUS):
        headpos = self.get_current_headpos(topic, uid, v_time)
        headmap = headpred.create_fixblur_map(headpos, tracesim_header.H, tracesim_header.W, tracesim_header.TILE_EXPAND_RADIUS)
        tilemap = headpred.expand(headmap, _radius)
        
        return tilemap
    
    def get_tilesizemap(self, topic, v_time):
        return self._tilesize_map_dict[topic][:, :, int(v_time), 1]
    
      
    def next_tile(self, topic, uid, pred_flag, v_time, model='lstm', _radius=tracesim_header.TILE_EXPAND_RADIUS):
        #given VTIME, predict for next chunk.
        #given VTIME, which part of the dataset should be selected?
        #print v_time
        #pred_time_pos, pred_time = self.match_time(topic, v_time)
        
        #print pred_time_pos, pred_time, v_time
        
        if pred_flag == True:
            v_time = v_time + self._look_ahead*0.063
            
            xi, yi = self.create_sample(topic, uid, v_time, self._look_back, self._look_ahead)
            fxi = self._lstm.predict(xi.reshape(1, *xi.shape))
            fxi2 = self._lnregr.predict(xi.reshape(1, -1))
            map_fxi3 = xi[-1][3:].reshape(tracesim_header.H, tracesim_header.W)

        
            map_fxi = headpred.create_fixblur_map(fxi[0], tracesim_header.H, tracesim_header.W, tracesim_header.TILE_EXPAND_RADIUS)
            map_fxi2 = headpred.create_fixblur_map(fxi2[0], tracesim_header.H, tracesim_header.W, tracesim_header.TILE_EXPAND_RADIUS)
            map_yi = headpred.create_fixblur_map(yi, tracesim_header.H, tracesim_header.W, tracesim_header.TILE_EXPAND_RADIUS)

            tile_fxi = headpred.expand(map_fxi, _radius)
            tile_fxi2 = headpred.expand(map_fxi2, _radius)
            tile_fxi3 = headpred.expand(map_fxi3, _radius)
            tile_fxi4 =  np.ones(shape=tile_fxi3.shape)
            
            tile_yi = headpred.expand(map_yi, tracesim_header.TILE_EXPAND_RADIUS)

            acc = headpred.overlap_accuracy(map_fxi, map_yi, _R=tracesim_header.TILE_EXPAND_RADIUS)
            acc2 = headpred.overlap_accuracy(map_fxi2, map_yi, _R=tracesim_header.TILE_EXPAND_RADIUS)
            acc3 = headpred.overlap_accuracy(map_fxi3, map_yi, _R=tracesim_header.TILE_EXPAND_RADIUS)
            
            acc = headpred.viewport_overlap(tile_yi, tile_fxi)
            acc2 = headpred.viewport_overlap(tile_yi, tile_fxi2)
            acc3 = headpred.viewport_overlap(tile_yi, tile_fxi3)
            acc4 = headpred.viewport_overlap(tile_yi, tile_fxi4)
            #print v_time, pred_flag, acc, acc2, acc3

            if model == self.LSTM:
                return tile_fxi, tile_yi, acc
            elif model == self.REGR:
                return tile_fxi2, tile_yi, acc2
            elif model == self.SAL:
                return tile_fxi3, tile_yi, acc3
            else:
                raise Exception
            
        else:
            tile_fxi = self.get_current_tilemap(topic, uid, v_time, _radius)
            return tile_fxi, tile_fxi, -1.0
    
    
       
        