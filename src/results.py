import matplotlib
import matplotlib.pyplot as plt
import socket
import sys
from os.path import isfile
from os.path import join

import numpy as np
from IPython.core.display import display

sys.path.append('../src')
import yaml


class DictTable(dict):
    # Overridden dict class which takes a dict in the form {'a': 2, 'b': 3},
    # and renders an HTML Table in IPython Notebook.
    def _repr_html_(self):
        html = ["<table border=0 width=100%>"]
        for key, value in self.iteritems():
            html.append("<tr>")
            html.append("<td>{0}</td>".format(key))
            html.append("<td>{0}</td>".format(value))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

class ListTable(list):
    # Overridden list class which takes a list in the form [1,2,3,4],
    # and renders an HTML Table in IPython Notebook.
    def _repr_html_(self):
        html = ["<table border=0 width=100%>"]
        html.append("<tr>")
        for val in self:
            html.append("<td><font size=\"0.1\">{0}</font></td>".format(val))
        html.append("</tr>")
        html.append("</table>")
        html.append("</p>")
        return ''.join(html)


fig_size = matplotlib.rcParams['figure.figsize']
matplotlib.rcParams['figure.figsize'] = (20.0, 4.0)
matplotlib.rcParams['savefig.dpi'] = 200
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['figure.dpi'] = 400


def prefix(i, zeros):
    s = str(i)
    while (len(s) < zeros):
        s = '0' + s
    return s


class Results(object):
    def __init__(self, date, experiment_number,local=False,experiment_group=''):

        if socket.gethostname() == 'ux305':
            if local:
                p = '/home/luka/Documents/autofaces'
            else:
                p = '/home/luka/v/lm1015-tmp'
        else:
            p = '/homes/lm1015/v/'

        if experiment_group != '':
             experiment_group += '_'

        path = join(join(p, 'data/' + date + '/'),experiment_group + prefix(experiment_number, 3))
        config_file = join(path, 'config.yaml')

        with open(config_file, 'r') as stream:
            try:
                self.config = yaml.load(stream)
            except yaml.YAMLError as exc:
                print exc

        self.au_map = self.config['data']['disfa_au_map']
        self.image_region_config = self.config['data'][self.config['data']['image_region']]
        if 'AUs' in self.image_region_config:
            aus = self.image_region_config['AUs']
            self.au_map = np.array(self.au_map,dtype=int)[aus]


        ssv_path = join(path, 'numerical_data')
        self.dash_settings = [8, 2]


        self.x_axis = np.loadtxt(join(ssv_path, 'x_axis.ssv'))
        self.lmsq_axis = np.loadtxt(join(ssv_path, 'lmsq.ssv'))
        self.cent_axis = np.loadtxt(join(ssv_path, 'cross_entropy.ssv'))
        self.accu_axis = np.loadtxt(join(ssv_path, 'naive_accuracy.ssv'))
        if 'autoencoder' in self.config and self.config['global']['use_autoencoder']:
            self.auto_axis = np.loadtxt(join(ssv_path, 'autoloss.ssv'))
            self.alpha_axis = np.loadtxt(join(ssv_path, 'alpha_axis.ssv'))
        self.per_au = np.load(join(ssv_path, 'per_au_accuracy.npz'))
        self.validation_auac_axis = self.per_au['validation_metrics']
        self.train_auac_axis = self.per_au['train_metrics']


        self.colors = matplotlib.cm.nipy_spectral(np.linspace(0,1,self.train_auac_axis.shape[1]))

        self.final_model = np.load(join(ssv_path, 'final_model_analysis.npz'))

        f = join(ssv_path, 'early_model_analysis.npz')
        if isfile(f):
            self.early_model = np.load(f)

    def print_config(self):
        d = self.config
        n = {}
        n['bias initial value'] = d['weights']['bias_start']
        n['weights initialisation'] = [d['weights']['weights_start_type'],
                                       d['weights']['weights_' + d['weights']['weights_start_type']]]
        d['weights'] = n

        image_region = d['data']['image_region']
        d['crop'] = d['data'][image_region]
        del d['data']['full']
        del d['data']['mouth']
        for key in d:
            if type(d[key]) == type({}):
                print key
                #                 print display(HTML('<b>'+'wat'+'</b>'))
                display(DictTable(d[key]))

    def au(self, prefix, smooth, filter_order=10, model=None):

        if prefix == 'train':
            x_axis = self.x_axis.copy()
            auac_axis = self.train_auac_axis.copy()
        elif prefix == 'test':
            if model == 'final':
                x_axis = self.final_model['threshold_values'].copy()
                auac_axis = self.final_model['test_threshold_data'].copy()
            elif model == 'early':
                x_axis = self.early_model['threshold_values'].copy()
                auac_axis = self.early_model['test_threshold_data'].copy()

        elif prefix == 'validation':
            x_axis = self.x_axis
            auac_axis = self.validation_auac_axis

        N, classes, calcs = auac_axis.shape
        # for i in xrange(classes):
        #     for j in xrange(calcs):
        #         for k in xrange(N):
        #             if np.isnan(auac_axis[k, i, j]):
        #                 auac_axis[k, i, j] = 0.0
        if smooth:
            for i in xrange(classes):
                for j in xrange(4):
                    first = auac_axis[0, i, j]
                    last = auac_axis[-1, i, j]
                    auac_axis[:, i, j] = np.convolve(auac_axis[:, i, j], np.ones(filter_order) / float(filter_order),
                                                     'same')

                    auac_axis[0, i, j] = first
                    auac_axis[-1, i, j] = last

        if prefix != 'test':
            f, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, sharex='col', sharey='row')
            axes = [ax1, ax2, ax3, ax4]
        else:
            f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex='col', sharey='row')
            axes = [ax1, ax2, ax3]

        titles = ['precision', 'recall', 'f1 score', 'area under roc']
        for i in xrange(classes):
            for j in xrange(len(axes)):
                axes[j].set_title(titles[j], fontsize=19)
                if (i+j) % 2:
                    dashes = [8, 2]
                else:
                    dashes = ''
                axes[j].plot(x_axis, auac_axis[:, i, j], label=str(self.au_map[i]),color=self.colors[i],dashes=dashes)
                axes[j].set_ylim(-0.1, 1.1)
                axes[j].set_xlim(-0.1, x_axis[-1] * 1.1)
                if prefix != 'test':
                    axes[j].set_xlabel('training iteration')
                else:
                    axes[j].set_xlabel('threshold')
        art = []
        lgd = plt.legend(loc=2, bbox_to_anchor=(1, 1), ncol=1)
        art.append(lgd)

        if prefix != 'test':
            labels = ax4.get_xticklabels()
            plt.setp(labels, rotation=30, fontsize=14)

        labels = ax3.get_xticklabels()
        plt.setp(labels, rotation=30, fontsize=14)

        labels = ax2.get_xticklabels()
        plt.setp(labels, rotation=30, fontsize=14)

        labels = ax1.get_xticklabels()
        plt.setp(labels, rotation=30, fontsize=14)

        if prefix == 'test':
            title = prefix + ' ' + model
        else:
            title = prefix
        plt.suptitle(title, fontsize=50, y=1.2)
        plt.show()

        if prefix == 'test':
            print 'Test ROC values for each class:'
            print '(values for p,r,f1 are maximums)'
            print 'class\troc\troc\tf1\tp\tr\t'
            for i in xrange(classes):
                roc = auac_axis[0, i, 3]
                f1 = round(auac_axis[:, i, 2].max(), 2)
                p = round(auac_axis[:, i, 0].max(), 2)
                r = round(auac_axis[:, i, 1].max(), 2)
                print self.au_map[i], '\t', round(roc, 2), '\t',
                if roc < 0.6:
                    print 'fail',
                elif roc < 0.7:
                    print 'poor',
                elif roc < 0.8:
                    print 'fair',
                elif roc < 0.9:
                    print 'good',
                else:
                    print 'great',
                print '\t', f1, '\t', p, '\t', r
            print 'average f1 = ', auac_axis[:, :, 2].max(axis=0).mean()
            print 'average roc = ', auac_axis[5, :, 3].mean()
            return
            print 'best thresholds'
            print 'au\tbest roc\tbest f1\t\tcol3-col2'
            for i in xrange(len(self.au_map)):
                t = self.final_model['test_roc_data'][0][i]
                if type(t) != type(None):
                    f = np.ones(len(t[0])) - t[0] + t[1]
                    c1 = round(t[2][f.argmax()], 2)
                    c2 = round(self.final_model['threshold_values'][auac_axis[:, i, 2].argmax()],2)
                    print self.au_map[i],'\t',c1,'\t\t',c2,'\t\t',c2-c1

    def get_f1_roc(self, prefix, model=None):

        if prefix == 'train':
            x_axis = self.x_axis.copy()
            auac_axis = self.train_auac_axis.copy()
        elif prefix == 'test':
            if model == 'final':
                x_axis = self.final_model['threshold_values'].copy()
                auac_axis = self.final_model['test_threshold_data'].copy()
            elif model == 'early':
                x_axis = self.early_model['threshold_values'].copy()
                auac_axis = self.early_model['test_threshold_data'].copy()

        elif prefix == 'validation':
            x_axis = self.x_axis
            auac_axis = self.validation_auac_axis

        return auac_axis[:, :, 2].max(axis=0).mean(), auac_axis[5, :, 3].mean()

    def get_excel_format(self):

        rocs = []

        x_axis = self.final_model['threshold_values']
        auac_axis = self.final_model['test_threshold_data']

        x_axis = self.early_model['threshold_values']
        auac_axis = self.early_model['test_threshold_data']


        for i in xrange(classes):
            rocs.append(auac_axis[0, i, 3])
        display(ListTable(rocs))


    def lmsq(self):
        plt.figure()
        plt.title('Classifer: mean squared error')
        plt.plot(self.x_axis, self.lmsq_axis[0, :], label='validation')
        plt.plot(self.x_axis, self.lmsq_axis[1, :], label='train')
        plt.ylim(0, 1)
        plt.legend()
        plt.show()
        # plt.savefig(join(save_path,'lmsq.png'),dpi=400)
        # plt.savefig(join(save_path_pdf,'lmsq.pdf'))

    def auto(self):
        if 'autoencoder' in self.config and self.config['global']['use_autoencoder']:
            plt.figure()
            plt.title('Autoencoder: mean squared error')
            plt.plot(self.x_axis, self.auto_axis[0, :], label='validation')
            plt.plot(self.x_axis, self.auto_axis[1, :], label='train')
            plt.plot(self.x_axis,self.alpha_axis,label='alpha')
            # plt.ylim(-.05, 1.05)
            # plt.ylim(-0.1, 1.1)
            plt.legend()
            plt.show()
            # plt.savefig(join(save_path,'lmsq.png'),dpi=400)
            # plt.savefig(join(save_path_pdf,'lmsq.pdf'))
        else:
            print 'No autoencoder'

    def cross(self):
        plt.figure()
        plt.title('Classifer: Cross entropy')
        # CUT = len(self.x_axis)/4
        CUT = 0
        plt.plot(self.x_axis[CUT:], self.cent_axis[0, CUT:], label='validation')
        plt.plot(self.x_axis[CUT:], self.cent_axis[1, CUT:], label='train')
        if (self.cent_axis < 1.0).sum() > 2:
            plt.ylim(0,1)
        plt.legend()
        # plt.ylim(-1.0,cent_axis.max()+1.0)
        plt.show()
        # plt.savefig(join(save_path,'cross_entropy.png'),dpi=400)
        # plt.savefig(join(save_path_pdf,'cross_entropy.pdf'))

    def test_confusion(self, model):

        if model == 'final':
            model_data = self.final_model
        elif model == 'early':
            model_data = self.early_model


        t = model_data['test_confusion']
        thresholds = model_data['threshold_values']
        test_threshold_data = model_data['test_threshold_data']

        for j in xrange(t[0].shape[0]):
            best_threshold = test_threshold_data[:, j, 2].argmax()
            roc = round(test_threshold_data[0, j, 3], 2)
            f1 = round(test_threshold_data[best_threshold, j, 2].max(), 2)
            p = round(test_threshold_data[best_threshold, j, 0].max(), 2)
            r = round(test_threshold_data[best_threshold, j, 1].max(), 2)
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Class ', self.au_map[j], 'threshold: ', round(thresholds[best_threshold],2)
            # print 'roc = ', roc,
            # print 'f1 = ', f1,
            # print 'p = ', p,
            # print 'r = ', r

            for x in xrange(2):
                print '\t[',
                for y in xrange(2):
                    print '',int(round(t[best_threshold, j, 1 - x, 1 - y],2)),'\t', #normalise / float(t[i, j, :, :].sum()), 2),
                print ']'
            print '\troc:\t',roc
            print '\tf1:\t',f1
            print '\tp:\t',p
            print '\tr:\t',r

    def roc_plot(self,model):
        if model == 'final':
            model_data = self.final_model
        elif model == 'early':
            model_data = self.early_model


        for i in xrange(len(model_data['test_roc_data'][0])):
            t = model_data['test_roc_data'][0][i]
            if type(t) != type(None):
                plt.xlabel('false positive rate')
                plt.ylabel('true positive rate')

                if i % 2:
                    dashes = self.dash_settings
                else:
                    dashes = ''
                plt.plot(t[0], t[1], label=str(self.au_map[i]),color=self.colors[i],dashes=dashes)

                plt.xlim(-0.1, 1.1)
                plt.ylim(-0.1, 1.1)
                plt.title('roc ' + model, fontsize=19)
        # print t[2][(t[0]).argmax()]
        plt.legend(loc=2, bbox_to_anchor=(1, 1), ncol=1)
        x = np.linspace(0, 1, 10)
        plt.plot(x, x)
        plt.show()

        for i in xrange(len(model_data['test_roc_data'][0])):
            t = model_data['test_roc_data'][0][i]
            if type(t) != type(None):
                f = np.ones(len(t[0])) - t[0] + t[1]

                if i % 2:
                    dashes = self.dash_settings
                else:
                    dashes = ''
                plt.plot(t[2], f, label=str(self.au_map[i]),color=self.colors[i],dashes=dashes)

                plt.xlabel('threshold')
                plt.ylabel('tpr + 1 - fpr')
                plt.xlim(-0.1, 1.1)
        plt.legend(loc=2, bbox_to_anchor=(1, 1), ncol=1)
        x = np.linspace(0, 1, 10)
        # plt.plot(x,x)
        plt.show()

    def print_auto_results(self):
        print 'Mean squared error for autoencoder on test set:'
        print 'final :', round(self.final_model['autoencoder_loss'],5),round(self.final_model['true_autoencoder_loss'],5)
        print 'early :', round(self.early_model['autoencoder_loss'],5),round(self.early_model['true_autoencoder_loss'],5)