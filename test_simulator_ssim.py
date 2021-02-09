import tracesim_header
import tracesim_buffer
import tracesim_network
import tracesim_predictor
import headpred
import train_lstm_headsal
import tracesim_simulate
import tracesim_bandwidth
import tracesim_psnr
import test_simulator_lib

reload(tracesim_network)
reload(tracesim_buffer)
reload(tracesim_predictor)
reload(tracesim_header)
reload(headpred)
reload(tracesim_simulate)
reload(tracesim_bandwidth)
reload(tracesim_psnr)

import pickle
import numpy as np

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


bandwidth, delay = 2000000, 0.2# 8000000, 0.04
delay = 0.04
look_back = 8
look_ahead = 16
tilesize_map_template = './dat/tilesize_map_{}'
sds_path = '../testing/salient_ds_dict_w16_h9'
model_lstm_path = './models/lstm_128128_lookahead{}_full_epo2000_newratio'
bwtrace_path = './4G_log/report_bus_0004.log'
bwtrace_path = './4G_log/report_bicycle_0001.log'
bwtrace_path = './3G_log/test1.txt'
#bwtrace_path = './3G_log/test2.txt'
sim = tracesim_simulate.Simulator(tilesize_map_template, sds_path, model_lstm_path, look_back, look_ahead, bwtrace_path, delay)



bwtrace_path.split('/')[-1].replace('.txt', '')
bwtrace_name = bwtrace_path.split('/')[-1].replace('.txt', '')

lstm_quallog = {}
lnregr_quallog = {}
sal_quallog = {}
full_quallog = {}

lstm_log = {}
lnregr_log = {}
sal_log = {}
full_log = {}

RAD = 4

#for RAD in [2, 3, 4, 5]:
    #bwscale = 50
bwtrace_list = ['./4G_log/report_car_0002.log',
                './4G_log/report_car_0003.log',
                './4G_log/report_car_0004.log',
                './4G_log/report_bus_0002.log',
                './4G_log/report_bus_0003.log',
                './4G_log/report_bus_0004.log',
               './4G_log/report_foot_0001.log']
modelname_list = ['lstm', 'lnregr', 'sal', 'full']
bw_std = 3.0
bw_meanlist = [ 4.0 ,  8.5, 13.0 , 17.5, 22.0 , 26.5, 31.0 ]
bwscalelist_str = '_'.join(map(str, bw_meanlist))
for scale_mean in bw_meanlist:#[1.0, 2.5, 5.5, 10.0, 15.0, 20.0]:
    scale_std = bw_std
    bwscale = scale_mean
    
    lstm_log[bwscale] = []
    lnregr_log[bwscale] = []
    sal_log[bwscale] = []
    full_log[bwscale] = []

    lstm_quallog[bwscale] = []
    lnregr_quallog[bwscale] = []
    sal_quallog[bwscale] = []
    full_quallog[bwscale] = []
    
    for bwtrace in bwtrace_list:
        sim._bandwidth = tracesim_bandwidth.Bandwidth(bwtrace, scale_mean=scale_mean, scale_std=scale_std)
        #sim._bandwidth.set_scale(bwscale)

        print 'running for bw{}'.format(bwscale)

        for TOPIC in ['0', '1', '2', '3', '4', '6', '7']:
            bw_bound = sim._pred._tilesize_map_dict[TOPIC][:, :, :, 1].sum()
            for UID in range(1, 8):
                print 'running for ', TOPIC, UID
                sim.run(TOPIC, UID, model_name=sim._pred.LSTM, radius=RAD)
                lstm_log[bwscale].append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME],\
                                sim._log[sim._BW_USED]*1.0/bw_bound, np.mean(sim._log[sim._ACC_LOG])])
                lstm_quallog[bwscale] += sim._log[sim._SSIM_LIST]

                sim.run(TOPIC, UID, model_name=sim._pred.REGR, radius=RAD)
                lnregr_log[bwscale].append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME],\
                                  sim._log[sim._BW_USED]*1.0/bw_bound, np.mean(sim._log[sim._ACC_LOG])])
                lnregr_quallog[bwscale] += sim._log[sim._SSIM_LIST]

                sim.run(TOPIC, UID, model_name=sim._pred.SAL, radius=RAD)
                sal_log[bwscale].append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME],\
                               sim._log[sim._BW_USED]*1.0/bw_bound, np.mean(sim._log[sim._ACC_LOG])])
                sal_quallog[bwscale] += sim._log[sim._SSIM_LIST]

                sim.run_full(TOPIC, UID)
                full_log[bwscale].append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME],\
                                sim._log[sim._BW_USED]*1.0/bw_bound, np.mean(sim._log[sim._ACC_LOG])])
                full_quallog[bwscale] += sim._log[sim._SSIM_LIST]
            
    lstm_log[bwscale] = np.array(lstm_log[bwscale])
    lnregr_log[bwscale] = np.array(lnregr_log[bwscale])
    sal_log[bwscale] = np.array(sal_log[bwscale])
    full_log[bwscale] = np.array(full_log[bwscale])

    #every bandwidth, save the whole information: what type of bw, rad, lookahead, scale
test_simulator_lib.save_log([lstm_log, lnregr_log, sal_log, full_log], bwtrace_name, RAD, look_ahead, bwscalelist_str)

#shuffle before write to disk
for quallog in [lstm_quallog, lnregr_quallog, sal_quallog, full_quallog]:
        for bwtrace in quallog:
            np.random.shuffle(quallog[bwtrace])    
#only retain top certain data (too many data)
limit=2000#only store first 1000 random data points
for quallog in [lstm_quallog, lnregr_quallog, sal_quallog, full_quallog]:
        for bwtrace in quallog:
            quallog[bwtrace] = quallog[bwtrace][:limit]
            
#output        

test_simulator_lib.save_quallog([lstm_quallog, lnregr_quallog, sal_quallog, full_quallog], modelname_list, RAD, look_ahead, bwscalelist_str)        




ssim_lstm, ssim_lnregr, ssim_sal, ssim_full = [], [], [], []
ssim_list = [[], [], [], []]
for idx, quallog in enumerate([lstm_quallog, lnregr_quallog, sal_quallog, full_quallog]):
    print 'generating for models'
    for bwtrace in sorted(quallog.keys()):
        ssim_list[idx] .append(test_simulator_lib.cal_psnr_log(quallog[bwtrace][:500], sim))

print ssim_list
test_simulator_lib.save_ssim(ssim_list, RAD, look_ahead, bwscalelist_str)
