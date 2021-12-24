import pickle
import numpy as np
import headpred

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})



def cal_psnr_log(_log, _sim):
    result = []
    for topic, ts, t1, t2 in _log:
        pr_pred, sm_pred, pr_full, sm_full, im0, im1, viewport_pred, viewport_full, viewport0 = _sim._psnr.viewport_perceived(topic, ts, t1, t2)
        #print topic, ts, pr_pred, sm_pred
        result.append([pr_pred, sm_pred])
    return np.array(result)

def load_log(filename, rad, look, bwscale):
    log_filepath = './result/' + filename
    return pickle.load(open('{}_rad{}_look{}_bwscale_{}'.format(log_filepath, rad, look, bwscale)))

def load_quallog(modelname_list, rad, look, bwscale):
    log_filepath = '/home/u9168/tracesim/result/QUALLOG'
    return pickle.load(open('{}_rad{}_look{}_bwscale_{}'.format(log_filepath, rad, look, bwscale)))

def load_ssim(rad, look, bwscale_str):
    log_filepath = './result/SSIM'
    return pickle.load(open('{}_rad{}_look{}_bwscale_{}'.format(log_filepath, rad, look, bwscale_str)))

def save_ssim(ssimlist, rad, look, bwscale_str):
    log_filepath = './result/SSIM'
    pickle.dump(ssimlist, open('{}_rad{}_look{}_bwscale_{}'.format(log_filepath, rad, look, bwscale_str), 'wb'))

def save_log(loglist, filename, rad, look, bwscale):
    log_filepath = '/home/u9168/tracesim/result/' + filename
    
    #pickle.dump([lstm_log, lnregr_log, sal_log, full_log], open('{}_rad{}_look{}_bwscale{}'.format(bwtrace_path, RAD, look_ahead, bwscale), 'wb'))
    pickle.dump(loglist, open('{}_rad{}_look{}_bwscale_{}'.format(log_filepath, rad, look, bwscale), 'wb'))
    print ('Stall count; stall_time; start_time, bw_used_ratio, non_blank_ratio')
    for llog in loglist:
        for item in llog:
            print (llog[item])#.mean(axis=0)

def save_quallog(quallog_list, modelname_list, rad, look, bwscale):
    log_filepath = '/home/u9168/tracesim/result/QUALLOG'
    pickle.dump(quallog_list, open('{}_rad{}_look{}_bwscale_{}'.format(log_filepath, rad, look, bwscale), 'wb'))

    for idx, quallog in enumerate(quallog_list):
        print ('Viewport overlap for model {}'.format( modelname_list[idx]))
        for bwtrace in quallog:
            print (bwtrace, np.mean([headpred.viewport_overlap(t1, t2) for topic, ts, t1, t2 in quallog[bwtrace]][:400]))
    return



def plot_log(plt, x, x_ticks, y_list, modelname_list, title='', xlabel='', ylabel='', axis=[]):
    plt.rcParams.update({'font.size': 15})
    plt.figure()
    marker_list = ['o', 'x', '+', '*']
    for idx, y in enumerate(y_list):
        plt.plot(x, y.T.mean(axis=0), linestyle='--', dashes=(5, 3), linewidth=2.45)
        
        plt.xticks(x, x_ticks, fontsize=20)
        plt.scatter(x, y.T.mean(axis=0), label=modelname_list[idx], s=180, marker=marker_list[idx], linewidths=2.45)
        plt.ylabel(ylabel, fontsize=20, fontweight='bold')
        plt.xlabel(xlabel, fontsize=20, fontweight='bold')
    plt.legend(prop={'size': 13})
    plt.tight_layout()
    plt.savefig('./figs/{}.pdf'.format(title), format='pdf', dpi=1000)
    
    fig=plt.figure(figsize=(11, 8), dpi= 80, facecolor='w', edgecolor='k')
    for idx, y in enumerate(y_list):
        plt.subplot(1, 4, idx+1)
        plt.title(modelname_list[idx])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.boxplot(y.T)
        plt.xticks(x+1, x_ticks)
        if len(axis) > 0: plt.axis(axis)
    
def plot_ssimlog():
    return