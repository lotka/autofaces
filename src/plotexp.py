import numpy as np
import matplotlib.pyplot as plt
import os
import sys
plt.style.use('bmh')

def prefix(i,zeros):
    s = str(i)
    while(len(s) < zeros):
        s = '0' + s
    return s

path = 'tbd/2016_05_30/' + prefix(sys.argv[1],3)
ssv_path = os.path.join(path,'numerical_data')

x_axis = np.loadtxt(os.path.join(ssv_path,'x_axis.ssv'))
lmsq_axis = np.loadtxt(os.path.join(ssv_path,'lmsq.ssv'))
cent_axis = np.loadtxt(os.path.join(ssv_path,'cross_entropy.ssv'))
accu_axis = np.loadtxt(os.path.join(ssv_path,'naive_accuracy.ssv'))
per_au = np.load(os.path.join(ssv_path,'per_au_accuracy.npz'))
test_auac_axis = per_au['test_metrics']
train_auac_axis = per_au['train_metrics']

plt.figure()
plt.title('Least mean squared error')
plt.plot(x_axis,lmsq_axis[0,:],label='test')
plt.plot(x_axis,lmsq_axis[1,:],label='train')
plt.ylim(-0.1,1.1)
plt.legend()
plt.savefig(os.path.join(path,'lmsq.png'),dpi=400)

plt.figure()
plt.title('Cross entropy')
plt.plot(x_axis,cent_axis[0,:],label='test')
plt.plot(x_axis,cent_axis[1,:],label='train')
plt.legend()
plt.ylim(-1.0,cent_axis.max()+1.0)
plt.savefig(os.path.join(path,'cross_entropy.png'),dpi=400)

plt.figure()
plt.title('Naive accuracy')
plt.plot(x_axis,accu_axis[0,:],label='test')
plt.plot(x_axis,accu_axis[1,:],label='train')
plt.ylim(-0.1,1.1)
plt.legend()
plt.savefig(os.path.join(path,'accuracy.png'),dpi=400)

def au(x_axis,auac_axis,prefix):

    N,classes,calcs = auac_axis.shape
    for i in xrange(classes):
        for j in xrange(calcs):
            for k in xrange(N):
                if np.isnan(auac_axis[k,i,j]):
                    auac_axis[k,i,j] = 0.0

    scaling_factor = 100
    if N > scaling_factor:
        nN = N/scaling_factor
        nxaxis = np.zeros(nN)
        smoothed = np.zeros((nN,classes,calcs))

        for i in xrange(classes):
            for j in xrange(calcs):
                for n in xrange(nN):
                    l = n*scaling_factor
                    r = (n+1)*scaling_factor
                    if r == N:
                        nxaxis[n] = x_axis[r-1]
                    else:
                        nxaxis[n] = x_axis[r]
                    smoothed[n,i,j] = auac_axis[l:r,i,j].mean()
        x_axis, auac_axis = (nxaxis, smoothed)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    axes = [ax1, ax2,ax3, ax4]
    titles = ['precision','recall','f1 score','area under roc']
    for i in xrange(classes):
        for j in xrange(len(axes)):
            axes[j].set_title(titles[j])
            axes[j].plot(x_axis, auac_axis[:,i,j],label=str(i))
            axes[j].set_ylim(-0.1,1.1)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(-0.1, -0.1), ncol=4)
    art.append(lgd)
    plt.ylim(-0.1,1.1)
    labels = ax4.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    labels = ax3.get_xticklabels()
    plt.suptitle(prefix)
    plt.setp(labels, rotation=30, fontsize=10)
    plt.savefig(
        os.path.join(path,prefix+'_per_au.png'),dpi=400, additional_artists=art,
        bbox_inches="tight")

au(x_axis,train_auac_axis,'train')
au(x_axis,test_auac_axis,'test')
