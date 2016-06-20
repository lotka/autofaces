#!/usr/bin/python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join
from os.path import isdir
from os import mkdir
import sys
# plt.style.use('bmh')

def prefix(i,zeros):
    s = str(i)
    while(len(s) < zeros):
        s = '0' + s
    return s

path = join('data/2016_06_15/',prefix(sys.argv[1],3))
ssv_path = join(path,'numerical_data')
save_path = join(path,'graphs')
save_path_pdf = join(save_path,'pdf')

if not isdir(save_path):
    mkdir(save_path)

if not isdir(save_path_pdf):
    mkdir(save_path_pdf)

x_axis = np.loadtxt(join(ssv_path,'x_axis.ssv'))
print join(ssv_path,'x_axis.ssv')
lmsq_axis = np.loadtxt(join(ssv_path,'lmsq.ssv'))
cent_axis = np.loadtxt(join(ssv_path,'cross_entropy.ssv'))
accu_axis = np.loadtxt(join(ssv_path,'naive_accuracy.ssv'))
per_au = np.load(join(ssv_path,'per_au_accuracy.npz'))
validation_auac_axis = per_au['validation_metrics']
train_auac_axis = per_au['train_metrics']
threshold_values = per_au['threshold_values']
test_threshold_data = per_au['test_threshold_data']

plt.figure()
plt.title('Least mean squared error')
plt.plot(x_axis,lmsq_axis[0,:],label='validation')
plt.plot(x_axis,lmsq_axis[1,:],label='train')
plt.ylim(-0.1,1.1)
plt.legend()
plt.savefig(join(save_path,'lmsq.png'),dpi=400)
plt.savefig(join(save_path_pdf,'lmsq.pdf'))

plt.figure()
plt.title('Cross entropy')
plt.plot(x_axis,cent_axis[0,:],label='validation')
plt.plot(x_axis,cent_axis[1,:],label='train')
plt.legend()
plt.ylim(-1.0,cent_axis.max()+1.0)
plt.savefig(join(save_path,'cross_entropy.png'),dpi=400)
plt.savefig(join(save_path_pdf,'cross_entropy.pdf'))

plt.figure()
plt.title('Naive accuracy')
plt.plot(x_axis,accu_axis[0,:],label='validation')
plt.plot(x_axis,accu_axis[1,:],label='train')
plt.ylim(-0.1,1.1)
plt.legend()
plt.savefig(join(save_path,'accuracy.png'),dpi=400)
plt.savefig(join(save_path_pdf,'accuracy.pdf'))

def au(x_axis,auac_axis,prefix,smooth):

    N,classes,calcs = auac_axis.shape
    for i in xrange(classes):
        for j in xrange(calcs):
            for k in xrange(N):
                if np.isnan(auac_axis[k,i,j]):
                    auac_axis[k,i,j] = 0.0

    # scaling_factor = 10
    # if True:
    #     nN = N/scaling_factor
    #     nxaxis = np.zeros(nN)
    #     smoothed = np.zeros((nN,classes,calcs))
    #
    #     for i in xrange(classes):
    #         for j in xrange(calcs):
    #             for n in xrange(nN):
    #                 l = n*scaling_factor
    #                 r = (n+1)*scaling_factor
    #                 if r == N:
    #                     nxaxis[n] = x_axis[r-1]
    #                 else:
    #                     nxaxis[n] = x_axis[r]
    #                 smoothed[n,i,j] = auac_axis[l:r,i,j].mean()
    #     x_axis, auac_axis = (nxaxis, smoothed)
    if smooth:
        for i in xrange(classes):
            for j in xrange(4):

                first = auac_axis[0,i,j]
                last = auac_axis[-1,i,j]

                auac_axis[:,i,j] = np.convolve(auac_axis[:,i,j],np.ones(10)/float(10), 'same')

                auac_axis[0,i,j] = first
                auac_axis[-1,i,j] = last


    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    axes = [ax1, ax2,ax3, ax4]
    titles = ['precision','recall','f1 score','area under roc']
    for i in xrange(classes):
        for j in xrange(len(axes)):
            axes[j].set_title(titles[j])
            axes[j].plot(x_axis, auac_axis[:,i,j],label=str(i))
            axes[j].set_ylim(-0.1,1.1)
            axes[j].set_xlim(-0.1,x_axis[-1]*1.1)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(-0.1, -0.1), ncol=4)
    art.append(lgd)
    labels = ax4.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    labels = ax3.get_xticklabels()
    plt.suptitle(prefix)
    plt.setp(labels, rotation=30, fontsize=10)
    plt.savefig(
        join(save_path,prefix+'_per_au.png'),dpi=400, additional_artists=art,
        bbox_inches="tight")
    plt.savefig(
        join(save_path_pdf,prefix+'_per_au.pdf'),additional_artists=art,
        bbox_inches="tight")

au(x_axis,train_auac_axis,'train',True)
au(x_axis,validation_auac_axis,'validation',True)
print threshold_values
au(threshold_values,test_threshold_data,'test',False)
