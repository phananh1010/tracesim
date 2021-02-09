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

import pickle
import numpy as np

reload(tracesim_network)
reload(tracesim_buffer)
reload(tracesim_predictor)
reload(tracesim_header)
reload(headpred)
reload(tracesim_simulate)
reload(tracesim_bandwidth)
reload(tracesim_psnr)
reload(test_simulator_lib)

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def create_model_sim(look_ahead):
    print 'CREATE SIM object with look ahead: {}'.format(look_ahead)
    bandwidth, delay = 2000000, 0.2# 8000000, 0.04
    delay = 0.04
    look_back = 8
    tilesize_map_template = './dat/tilesize_map_{}'
    sds_path = '../testing/salient_ds_dict_w16_h9'
    model_lstm_path = './models/lstm_128128_lookahead{}_full_epo2000_ratio0024_sgd'
    bwtrace_path = './4G_log/report_bus_0004.log'
    bwtrace_path = './4G_log/report_bicycle_0001.log'
    bwtrace_path = './3G_log/test1.txt'
    #bwtrace_path = './3G_log/test2.txt'
    sim = tracesim_simulate.Simulator(tilesize_map_template, sds_path, model_lstm_path, look_back, look_ahead, bwtrace_path, delay)
    
    return sim

def run_model_sim(sim, RAD_list, bwscale):

    RAD = 3
    TOPIC = '3'
    UID = 1


    lstm_log = {}
    lnregr_log = {}
    sal_log = {}
    full_log = {}


    sim._bandwidth.set_scale(bwscale)
    for RAD in RAD_list:
        lstm_log[RAD] = []
        lnregr_log[RAD] = []
        sal_log[RAD] = []
        full_log[RAD] = []

        #print 'running for bw{}'.format(bwscale)
        print 'running for rad {}'.format(RAD)

        for TOPIC in ['0', '1', '2', '3', '4', '6', '7', '8']:
            bw_bound = sim._pred._tilesize_map_dict[TOPIC][:, :, :, 1].sum()
            for UID in range(1, 10):
                print 'running for ', TOPIC, UID
                sim.run(TOPIC, UID, model_name=sim._pred.LSTM, radius=RAD)
                lstm_log[RAD].append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME],\
                                sim._log[sim._BW_USED]*1.0/bw_bound, np.mean(sim._log[sim._ACC_LOG])])
                #lstm_quallog[bwscale] += sim._log[sim._SSIM_LIST]

                sim.run(TOPIC, UID, model_name=sim._pred.REGR, radius=RAD)
                lnregr_log[RAD].append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME],\
                                  sim._log[sim._BW_USED]*1.0/bw_bound, np.mean(sim._log[sim._ACC_LOG])])
                #lnregr_quallog[bwscale] += sim._log[sim._SSIM_LIST]

                sim.run(TOPIC, UID, model_name=sim._pred.SAL, radius=RAD)
                sal_log[RAD].append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME],\
                               sim._log[sim._BW_USED]*1.0/bw_bound, np.mean(sim._log[sim._ACC_LOG])])
                #sal_quallog[bwscale] += sim._log[sim._SSIM_LIST]

                sim.run_full(TOPIC, UID)
                full_log[RAD].append([sim._log[sim._STALL_COUNT], sim._log[sim._STALL_TIME] - sim._log[sim._STALL_INIT_TIME], sim._log[sim._STALL_INIT_TIME],\
                                sim._log[sim._BW_USED]*1.0/bw_bound, np.mean(sim._log[sim._ACC_LOG])])
                #full_quallog[bwscale] += sim._log[sim._SSIM_LIST]

        lstm_log[RAD] = np.array(lstm_log[RAD])
        lnregr_log[RAD] = np.array(lnregr_log[RAD])
        sal_log[RAD] = np.array(sal_log[RAD])
        full_log[RAD] = np.array(full_log[RAD])
    return [lstm_log, lnregr_log, sal_log, full_log]

filemark = 'modellog'
for look_ahead in [8, 16, 24, 32]:
    simi = create_model_sim(look_ahead)
    RAD_list = [2, 3, 4, 5]
    radlist_str = '_'.join(map(str, RAD_list))
    bwscale = 5.0
    lstm_log, lnregr_log, sal_log, full_log = run_model_sim(simi, RAD_list, bwscale)
    loglist = (lstm_log, lnregr_log, sal_log, full_log)
    test_simulator_lib.save_log(loglist,filemark , radlist_str, look_ahead, bwscale)

