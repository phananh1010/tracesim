import tracesim_header
import copy

class NetworkEvent:
    #this class is a timer, 
    #begin event will reset timer, reset state
    #state: idle (at the begining), requesting, receiving, done (waiting for next request)
    
    def __init__(self, bandwidth, net_delay):
        self._bandwidth = bandwidth
        self._net_delay = net_delay
        self._status = tracesim_header.NETWORK_STATUS_IDLE 
        
        self._request_time = 0.0
        self._receive_time = 0.0
        self._timer = 0.0
        
        self._vid_time = 0.0
        self._tile_map = None

    def getState(self):
        #move to next state. the network event shoudl call this after it has finished something
        if self._request_time == 0.0 and self._receive_time == 0.0:
            return tracesim_header.NETWORK_STATUS_IDLE
        else:
            if self._timer <= 0:
                return tracesim_header.NETWORK_STATUS_DONE
            elif self._timer < self._receive_time:
                return tracesim_header.NETWORK_STATUS_RECEIVING
            elif self._timer <= self._receive_time + self._request_time:
                return tracesim_header.NETWORK_STATUS_REQUESTING
            else:
                print 'ERROR: timer: {}'.format
                raise #should not be here
            
    def startRequest(self, g_time, vid_time, tile_map, tilesize_map):
        #calculate the timer for request based on network delay. 
        #set request timer = 20 miliseconds = 0.02
        self._request_time = self._net_delay #based on delay & bandwidth
        data_size = tilesize_map[tile_map > 0].sum()
        self._receive_time = self._net_delay +  data_size*1.0 / self._bandwidth.get(g_time) #based on delay & bandwidth & total tiles size
        self._timer = self._request_time + self._receive_time
        self._vid_time = vid_time
        self._tile_map = tile_map
        
    def endRequest(self):
        self._request_time = 0.0
        self._receive_time = 0.0
        self._timer = 0.0
        
        temp = self._vid_time
        temp2 = copy.deepcopy(self._tile_map)
        
        self._vid_time = 0
        self._tile_map = None
        return temp, temp2
    
    def isReady(self):
        if self.getState() == tracesim_header.NETWORK_STATUS_IDLE:
            return True
        else:
            return False
        
    def isFinished(self):
        if self.getState() == tracesim_header.NETWORK_STATUS_DONE:
            return True
        else:
            return False
    
    def proceed(self, step): 
        if self._timer <= -step/2:
            #this make sure the scheduler must be responsible and check the network to pump data to buffer
            raise
        if self._timer > 0:
            self._timer -= step
        return self.getState() #let the scheduler know what state the network is on
        