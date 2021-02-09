from scipy.interpolate import interp1d
import numpy as np

class Bandwidth:
    _trace_filepath = ''#'./4G_log/report_bus_0002.log'
    _trace = []
    _scale = 1.0
    def __init__(self, bwtrace_filepath, scale_mean=-1.0, scale_std=-1.0):
        self._trace_filepath = bwtrace_filepath
        self._trace = self.load(self._trace_filepath)
        self._trace = self.scale_bw(self._trace, scale_mean, scale_std)
        self._trace = self._trace[10:]
        
    def scale_bw(self, bw, m=-1.0, v=-1.0):#m~scale, v~variance
        #self._trace is byte per seconds. input m, v is Mbps
        #bring bw to mean= m Mbps, std=v 
        #bw input is bytes per seconds, must convert it into Mbps, then convert it back to bytes
        lowerbound = 1.5
        
        temp = np.array(bw)*8.0/1000000 #convert to mbps
        if v < 0: 
            v = temp.std()
        if m < 0:
            m = temp.mean()
        temp = (temp - temp.mean())/temp.std()
        temp = temp * v + m
        temp[temp<lowerbound] = lowerbound
        result = np.array(temp * 1000000.0/ 8.0, dtype=int)
        if (result<0).sum() > 0:
            print 'Bandwidth < 0 after scaling ({}-{}) mean:{}, min:{}, max:{}'.format(m, v, result.mean(), result.min(), result.max())
            raise Exception
        return result.tolist()
        
    def load(self, input_path):
        dat_raw = open(input_path).read().split('\n')[:-1]
        dat = np.array([[float(item.split(' ')[1])*0.001, float(item.split(' ')[-2])] for item in dat_raw])
        
        dat[:, 0] = dat[:, 0] - dat[0][0]
        #print dat
        x = np.arange(dat[0][0], dat[-1][0], 1)
        f = interp1d(dat[:, 0], dat[:, 1])
        fx = np.array(f(x), dtype=int)
        
        return fx.tolist()
    
    def get(self, vtime):
        return int(self._trace[int(vtime%len(self._trace))] * self._scale)
    
    def get_info(self):
        #return bandwidth mean and variance in Mbps
        temp = np.array(bw._trace)*8.0/1000000 #convert to mbps
        return temp.mean(), temp.std()
    
    def set_scale(self, new_scale):
        self._scale = new_scale