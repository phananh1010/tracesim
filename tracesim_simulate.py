import numpy as np

import tracesim_header
import tracesim_predictor
import tracesim_network
import tracesim_buffer
import tracesim_bandwidth
import tracesim_psnr
import headpred

class Simulator:
    _look_back = 0
    _look_ahead = 0
    
    _tilesize_map_path = ''#'./dat/tilesize_map_{}'
    _sds_path = ''#'../testing/salient_ds_dict_w16_h9'
    _model_lstm_path = ''#'./models/lstm_128128_lookahead{}_full_epo2000_newratio'
    _bandwidth = None #load from './4G_log/report_bus_0002.log'
    _delay = 0.0
    
    _psnr = None
    _pred = None
    _buff = None
    _network = None
    
    _NET_LOG = 'NET_LOG'
    _ACC_LOG = 'ACC_LOG'
    _STALL_TIME = 'STALL_TIME'
    _STALL_COUNT = 'STALL_COUNT'
    _STALL_INIT_TIME = 'STALL_INIT_TIME'
    _BW_USED = 'BW_USED'
    _BLANK_RATIO = 'BLANK_RATIO'
    _SSIM = 'SSIM'
    _FULL_SSIM = 'FULL_SSIM'
    _SSIM_LIST = 'SSIM_LIST'
    _log = {_NET_LOG: [], _ACC_LOG: [], _STALL_TIME: 0, _STALL_COUNT: 0, _STALL_INIT_TIME: 0, _BW_USED:[]}
    
    def __init__(self, tilesize_map_path, sds_path, model_lstm_path, look_back, look_ahead, bwtrace_filepath, delay):
        self._look_back = look_back
        self._look_ahead = look_ahead
        self._tilesize_map_path = tilesize_map_path
        self._sds_path = sds_path
        self._model_lstm_path = model_lstm_path
        self._bandwidth = tracesim_bandwidth.Bandwidth(bwtrace_filepath)
        self._delay = delay
        self._psnr = tracesim_psnr.PSNR()
        
        self.load_model()
        self.load_system()
        
    def load_model(self):
        #load prediction model
        self._pred =  tracesim_predictor.Predictor(self._tilesize_map_path, self._sds_path, self._model_lstm_path,
                                    self._look_back, self._look_ahead)
    def load_system(self):
        #load system buffer and system network simulator
        
        return 
        #buff_state = tracesim_header.STATUS_BUFFERING
    
    def run(self, topic, uid, model_name, radius=2):#lstm, regr, sal
        self._log = {self._NET_LOG: [], self._ACC_LOG: [], self._STALL_TIME: 0, self._STALL_COUNT: 0, self._STALL_INIT_TIME: 0, self._BW_USED:0, self._SSIM_LIST: []}
        
        VTIME = self._pred.get_VTIME_0(topic)    
        GTIME = 0.0    
        buff_state = tracesim_header.STATUS_BUFFERING
        STEP = tracesim_header.STEP

        use_prediction = True
        
        
        self._network = tracesim_network.NetworkEvent(self._bandwidth, self._delay)#(8000000, 0.04)
        self._buff = tracesim_buffer.Buffer(3.0, 1.0, VTIME)#size, streaming thress
        
        for i in xrange(9999999999999):
            if VTIME >= self._pred.get_VTIME_N(topic)-1:
                break
            
            prev_buff_state = buff_state
            buff_state = self._buff.consume(STEP, buff_state)
            net_state = self._network.proceed(STEP)
            
            if self._buff.isEmpty() == True:
                viewport_tilemap = self._pred.get_current_tilemap(topic, uid, VTIME)
                pred_tilemap = np.zeros(shape=viewport_tilemap.shape)
            else:
                viewport_tilemap = self._pred.get_current_tilemap(topic, uid, VTIME)
                pred_tilemap = self._buff._content[0] #head of the buffer, to be pop
            self._log[self._SSIM_LIST].append((topic, VTIME, viewport_tilemap, pred_tilemap))
            
            #after consuming, see if user actually loooking at correct predicted viewport
            vp = self._pred.get_current_tilemap(topic, uid, VTIME)
            if self._buff.isEmpty() == True:
                self._log[self._ACC_LOG].append(0.0)
            else:
                pd = self._buff._content[0]
                self._log[self._ACC_LOG].append(headpred.viewport_overlap(vp, pd))
            
            if self._buff.isEmpty() == True:
                use_prediction = False
            elif self._buff.checkStalling(self._pred.get_current_tilemap(topic, uid, VTIME)) == True:
                use_prediction = False
                self._buff.clear()
                #print 'WARNING! MISPREDICTION, CLEAR BUFFER', VTIME
            else:
                use_prediction = True

            #now decide if need to send a network request
                #if network is free, and buffer is not over capacity
            if self._network.isReady() == True and self._buff.isReady() == True:
                #predict head position in next VTIME + look_ahead seconds
                fx, yi, acc = self._pred.next_tile(topic, uid, use_prediction, VTIME, model=model_name, _radius=radius)#lstm, lnregr, sal
                #however, request the tilemap after the buffer, maybe much more than look_ahead
                tilesizemap = self._pred.get_tilesizemap( topic, VTIME)
                self._network.startRequest(GTIME, self._buff._tail, fx, tilesizemap)
                
                self._log[self._BW_USED] += tilesizemap[fx>0].sum()
                #if acc != -1: 
                #    self._log[self._ACC_LOG].append(acc)

            if self._network.isFinished() == True:
                vid_time, fx = self._network.endRequest()
                #self, vid_time, chunk_size, tile_map, step
                self._buff.fill(vid_time, 1.0, fx, STEP)

            GTIME += STEP
            VTIME = self._buff._head
            self._log[self._NET_LOG].append([GTIME, VTIME, self._buff.getLength()])
                

            if buff_state == tracesim_header.STATUS_BUFFERING:
                self._log[self._STALL_TIME] += STEP
                if self._log[self._STALL_COUNT] == 0:
                    self._log[self._STALL_INIT_TIME] += STEP
                
            if prev_buff_state != tracesim_header.STATUS_BUFFERING and buff_state == tracesim_header.STATUS_BUFFERING:
                self._log[self._STALL_COUNT] += 1

        #net_log = np.array(net_log)
        self._log[self._NET_LOG] = np.array(self._log[self._NET_LOG])
        
        
    def run_full(self, topic, uid):#lstm, regr, sal
        model_name = self._pred.LSTM
        radius = 2
        self._log = {self._NET_LOG: [], self._ACC_LOG: [], self._STALL_TIME: 0, self._STALL_COUNT: 0, self._STALL_INIT_TIME: 0, self._BW_USED:0, self._SSIM_LIST: []}
        
        VTIME = self._pred.get_VTIME_0(topic)    
        GTIME = 0.0    
        buff_state = tracesim_header.STATUS_BUFFERING
        STEP = tracesim_header.STEP
        
        use_prediction = False
        self._network = tracesim_network.NetworkEvent(self._bandwidth, self._delay)#(8000000, 0.04)
        self._buff = tracesim_buffer.Buffer(3.0, 1.0, VTIME)#size, streaming thress
        
        for i in xrange(9999999999999):
            if VTIME >= self._pred.get_VTIME_N(topic)-1:
                break
            
            prev_buff_state = buff_state
            buff_state = self._buff.consume(STEP, buff_state)
            net_state = self._network.proceed(STEP)

            if self._buff.isEmpty() == True:
                viewport_tilemap = self._pred.get_current_tilemap(topic, uid, VTIME)
                pred_tilemap = np.zeros(shape=viewport_tilemap.shape)
            else:
                viewport_tilemap = self._pred.get_current_tilemap(topic, uid, VTIME)
                pred_tilemap = self._buff._content[0] #head of the buffer, to be pop
            self._log[self._SSIM_LIST].append((topic, VTIME, viewport_tilemap, pred_tilemap))
            
            #after consuming, see if user actually loooking at correct predicted viewport
            vp = self._pred.get_current_tilemap(topic, uid, VTIME)
            if self._buff.isEmpty() == True:
                self._log[self._ACC_LOG].append(0.0)
            else:
                pd = self._buff._content[0]
                self._log[self._ACC_LOG].append(headpred.viewport_overlap(vp, pd))
            
            if self._buff.isEmpty() == True:
                use_prediction = False
            elif self._buff.checkStalling(vp) == True:
                use_prediction = False
                self._buff.clear()
                print 'Why full download has viewing stalling'
                raise Exception
            else:
                use_prediction = True

            #now decide if need to send a network request
                #if network is free, and buffer is not over capacity
            if self._network.isReady() == True and self._buff.isReady() == True:
                #predict head position in next VTIME + look_ahead seconds
                fx, yi, acc = self._pred.next_tile(topic, uid, use_prediction, VTIME, model=model_name, _radius=radius)#lstm, lnregr, sal
                fx = np.ones(shape=yi.shape)
                #however, request the tilemap after the buffer, maybe much more than look_ahead
                tilesizemap = self._pred.get_tilesizemap( topic, VTIME)
                self._network.startRequest(GTIME, self._buff._tail, fx, tilesizemap)
                
                self._log[self._BW_USED] += tilesizemap[fx>0].sum()
                #if acc != -1: 
                #    self._log[self._ACC_LOG].append(acc)

            if self._network.isFinished() == True:
                vid_time, fx = self._network.endRequest()
                #self, vid_time, chunk_size, tile_map, step
                self._buff.fill(vid_time, 1.0, fx, STEP)

            GTIME += STEP
            VTIME = self._buff._head
            self._log[self._NET_LOG].append([GTIME, VTIME, self._buff.getLength()])
                

            if buff_state == tracesim_header.STATUS_BUFFERING:
                self._log[self._STALL_TIME] += STEP
                if self._log[self._STALL_COUNT] == 0:
                    self._log[self._STALL_INIT_TIME] += STEP
                
            if prev_buff_state != tracesim_header.STATUS_BUFFERING and buff_state == tracesim_header.STATUS_BUFFERING:
                self._log[self._STALL_COUNT] += 1

        #net_log = np.array(net_log)
        self._log[self._NET_LOG] = np.array(self._log[self._NET_LOG])