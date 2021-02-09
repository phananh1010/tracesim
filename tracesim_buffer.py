import tracesim_header
import numpy as np
import headpred

from matplotlib import pyplot as plt


class Buffer:
    _head = -1.0 #in seconds, display to user
    _tail = -1.0   #in seconds, append from network
    _size = -1.0  #size of buffer, stop download if excess
    _buffering_threshold = 0.0  #if there are more data in buffer than this threshold, move from buffering
                                   #to streaming
        
    _BUFFER_POS_ZERO = 0   #buffer is empty
    _BUFFER_POS_ZERO_TH = 1#have content, buf less than streaming threshold
    _BUFFER_POS_TH_SIZE = 2#have content, more than streaming threshold, less than size
    _BUFFER_POS_SIZE = 3   #have content, more than needed
    
    _next_status_dict = {\
        (_BUFFER_POS_ZERO, tracesim_header.STATUS_BUFFERING):tracesim_header.STATUS_BUFFERING,\
        (_BUFFER_POS_ZERO, tracesim_header.STATUS_STREAMING):tracesim_header.STATUS_BUFFERING,\
        (_BUFFER_POS_ZERO, tracesim_header.STATUS_EXCESS):tracesim_header.STATUS_BUFFERING,\
                         
        (_BUFFER_POS_ZERO_TH, tracesim_header.STATUS_BUFFERING):tracesim_header.STATUS_BUFFERING,\
        (_BUFFER_POS_ZERO_TH, tracesim_header.STATUS_STREAMING):tracesim_header.STATUS_STREAMING,\
        (_BUFFER_POS_ZERO_TH, tracesim_header.STATUS_EXCESS):tracesim_header.STATUS_BUFFERING,\
                         
        (_BUFFER_POS_TH_SIZE, tracesim_header.STATUS_BUFFERING):tracesim_header.STATUS_STREAMING,\
        (_BUFFER_POS_TH_SIZE, tracesim_header.STATUS_STREAMING):tracesim_header.STATUS_STREAMING,\
        (_BUFFER_POS_TH_SIZE, tracesim_header.STATUS_EXCESS):tracesim_header.STATUS_STREAMING,
                         
        (_BUFFER_POS_SIZE, tracesim_header.STATUS_BUFFERING):tracesim_header.STATUS_EXCESS,\
        (_BUFFER_POS_SIZE, tracesim_header.STATUS_STREAMING):tracesim_header.STATUS_EXCESS,\
        (_BUFFER_POS_SIZE, tracesim_header.STATUS_EXCESS):tracesim_header.STATUS_EXCESS,
         }
    
    _content = []
    
    def __init__(self, size, buffering_threshold, start_vid_time):
        
        self._head = start_vid_time
        self._tail = self._head
        self._size = size
        self._buffering_threshold = buffering_threshold
        self._content = []
        
    
    def getBufferPos(self):
        #given current buffer size, return if buffer is empty, less than threshold,\
        #less than size, or greater than size
        buffer_length = self.getLength()
        if buffer_length == 0:
            return self._BUFFER_POS_ZERO
        elif buffer_length < self._buffering_threshold:
            return self._BUFFER_POS_ZERO_TH
        elif buffer_length < self._size:
            return self._BUFFER_POS_TH_SIZE
        else:#greater or equal to the size
            return self._BUFFER_POS_SIZE
        
    def getLength(self):
        return self._tail - self._head
    
    def nextState(self, prev_status):
        buffer_length = self.getLength()
        buffPos = self.getBufferPos()
        return self._next_status_dict[(buffPos, prev_status)]
    
    def isEmpty(self):
        if len(self._content) == 0:
            return True
        else:
            return False
    
    def isReady(self):
        buffer_length = self._tail - self._head
        if buffer_length < self._size:
            return True
        else:
            return False
        
    def checkStalling(self, tile_map):
        #plt.imshow(tile_map)
        #plt.figure()
        #plt.imshow(self._content[0])
        #plt.figure()
        #print 'tile_map: ', tile_map
        #print 'buff: ', self._content[0]
        #print 'overlsap: ', headpred.viewport_overlap( tile_map, self._content[0])
        #print 'content: ',self._content[0].sum()
        #print 'overlap: ', np.array(np.logical_and(self._content[0], tile_map), dtype=int).sum()/self._content[0].sum()
        if headpred.viewport_overlap(tile_map, self._content[0]) > 0.75:
            return False
        else:
            return True
        
    def clear(self):
        self._tail = self._head
        self._content = []
        
    def consume(self, step, prev_status): #entry function for the simulation framework
        if prev_status != tracesim_header.STATUS_BUFFERING:
            #if it is buffering and size is less than threshold, no reason to consume it
            self._head = self._head + step
            self._content = self._content[1:]
        if self._head > self._tail:
            self._head = self._tail
        
        self._head = round(self._head, 2)
        self._tail = round(self._tail, 2)
        #print 'BUFFER CONSUME: ', self._head, self._tail, len(self._content), prev_status
        return self.nextState(prev_status)
    
    def fill(self, vid_time, chunk_size, tile_map, step): #take no time
        #print 'DEBUG: tail, head, tail-head, content, added', self._tail, self._head, (self._tail - self._head)*100, len(self._content), len(np.arange(vid_time*100, vid_time*100+chunk_size*100, round(step*100, 0))/100)
        
        self._tail += chunk_size
        #for i in np.arange(vid_time*100, vid_time*100+chunk_size*100, round(step*100, 0))/100:
        #    self._content.append(tile_map)
        
        pfrom = int(round(vid_time*100, 0))
        pto = int(round(vid_time*100 + chunk_size*100, 0))
        pstep = int(round(step * 100, 0))
        for i in np.arange(pfrom, pto, pstep):
            self._content.append(tile_map)
        
        if not np.allclose((self._tail - self._head)*100, len(self._content)):
            print 'BUFFER CONTENT must be EQUAL tail - head'
            print "len(self._content), self._tail, self._head, vid_time, vid_time+chunk_size, chunk_size"
            print len(self._content), self._tail, self._head, vid_time, vid_time+chunk_size, chunk_size
            raise
            
        #print 'BUFFER FILL: ', self._head, self._tail, len(self._content)
        #    
    def next_chunk_time(vid_time):
        if self._tail < vid_time:
            raise
        