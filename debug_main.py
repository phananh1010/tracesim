import tracesim_header
import tracesim_buffer
import tracesim_network
import tracesim_predictor
import headpred
import train_lstm_headsal
import tracesim_simulate

import pickle
import numpy as np

from matplotlib import pyplot as plt
#%matplotlib inline

reload(tracesim_network)
reload(tracesim_buffer)
reload(tracesim_predictor)
reload(tracesim_header)
reload(headpred)
reload(tracesim_simulate)

look_back = 8
look_ahead = 16
tilesize_map_template = './dat/tilesize_map_{}'
sds_path = '../testing/salient_ds_dict_w16_h9'
model_lstm_path = './models/lstm_128128_lookahead{}_full_epo2000_newratio'
bwtrace_path = './4G_log/report_bus_0002.log'
delay = 0.01
sim = tracesim_simulate.Simulator(tilesize_map_template, sds_path, model_lstm_path, look_back, look_ahead, bwtrace_path, delay)
RAD = 2
TOPIC = '2'
UID = 1

sim._bandwidth._trace = sim._bandwidth._trace[0:]

lstm_log = []
lnregr_log = []
sal_log = []
full_log = []
for TOPIC in ['0', '1', '2', '3', '4', '6', '7']:
    for UID in range(1, 3):
        print 'running for ', TOPIC, UID
        sim.run(TOPIC, UID, model_name=sim._pred.LSTM, radius=RAD)
        lstm_log.append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME]])
        
        sim.run(TOPIC, UID, model_name=sim._pred.REGR, radius=RAD)
        lnregr_log.append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME]])
        
        sim.run(TOPIC, UID, model_name=sim._pred.SAL, radius=RAD)
        sal_log.append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME]])
        
        sim.run_full(TOPIC, UID)
        full_log.append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME]])
        
lstm_log = np.array(lstm_log)
lnregr_log = np.array(lnregr_log)
sal_log = np.array(sal_log)
full_log = np.array(full_log)

print lstm_log.mean(axis=0), lnregr_log.mean(axis=0), sal_log.mean(axis=0), full_log.mean(axis=0)