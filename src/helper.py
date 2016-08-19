from itertools import product
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker

def plot_images(images,names=None,cmap='Spectral',interpolation='none',title=None,range=None,save=None):
    #wtf
    if names is None:
        names = ['image_'+str(i) for i in xrange(len(images))]

    fig_size = matplotlib.rcParams['figure.figsize']
    matplotlib.rcParams['figure.figsize'] = (20.0, 4.0)
    matplotlib.rcParams['savefig.dpi'] = 200
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['figure.dpi'] = 400

    fig = plt.figure()
    if title != None:
        fig.suptitle(title,fontsize=30)

    cmap = 'Spectral'

    for i,image in enumerate(images):
        plt.subplot(1,len(images),i+1)
        if range is None:
            plt.imshow(images[i].round(5)),interpolation='none',cmap=cmap)
        else:
            plt.imshow(images[i], interpolation='none', cmap=cmap,vmin=range[0],vmax=range[1])
        plt.title(names[i])
        cbar = plt.colorbar(orientation='horizontal')

        tick_locator = ticker.MaxNLocator(nbins=4)
        cbar.locator = tick_locator
        cbar.update_ticks()

        # ticks = cbar.ax.get_xticks()
        # cbar.ax.set_xticklabels(np.linspace(round(images[i].min(),1),round(images[i].max(),1),4), rotation=90)
        # if images[i].sum() == 0:
        #     ticks=None
        # else:
        #     ticks = [images[i].min(), 0, images[i].max()]
        # if range != None:
        #     ticks = [range[0],0,range[1]]
        # ticks = [images[i].min(), 0, images[i].max()]
        # for i,t in enumerate(ticks):
        #     ticks[i] = round(t,10)
        # cbar = plt.colorbar(ticks=ticks,orientation='horizontal')
        # cbar.ax.set_xticklabels(ticks, rotation=90)
    if save != None:
        # plt.tight_layout()
        print 'save to', save
        plt.savefig(save,bbox_inches='tight', pad_inches=0.1)
    else:
        print 'not saving nothing for you mate'
    plt.show()

def plot_lines(lines,names=None,labels=None,title=None,ylim=None,save=None):

    if names is None:
        names = [str(i) for i in xrange(len(lines))]

    fig_size = matplotlib.rcParams['figure.figsize']
    matplotlib.rcParams['figure.figsize'] = (20.0, 6.0)
    matplotlib.rcParams['savefig.dpi'] = 400
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['figure.dpi'] = 400
    matplotlib.rcParams['text.usetex'] = False
    # matplotlib.rcParams['font.family'] ='serif'

    fig = plt.figure()
    if title != None:
        fig.suptitle(title,fontsize=30)

    for i,line in enumerate(lines):
        x,ys = line
        plt.subplot(1,len(lines),i+1)
        plt.title(names[i])
        for i,y in enumerate(ys):
            if labels is None:
                plt.plot(x,y,label=None)
            else:
                plt.plot(x, y, label=labels[i])
            if ylim != None:
                plt.ylim(ylim[0],ylim[1])
    horz = -0.8
    vert = -.3
    lgd = plt.legend(loc='lower center', bbox_to_anchor=(horz, vert), ncol=len(lines))

    if save != None:
        # plt.tight_layout()
        plt.savefig(save,bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return

def get_all_experiments(bounds):
    """
    This function returns a list of overwrite dictionaries for use
    in the dragonfly brain
    Args:
        bounds (dict) : this defines the parameters you want to search over
        example:
        >>> bounds = {'type' : ['a','b'], 'value' : np.linspace(0,1,3)}
        >>> get_all_experiments(bounds)
        [{'type': 'a', 'value': 0.0},
         {'type': 'a', 'value': 0.5},
         {'type': 'a', 'value': 1.0},
         {'type': 'b', 'value': 0.0},
         {'type': 'b', 'value': 0.5},
         {'type': 'b', 'value': 1.0}]

    Return (dict) : list of overwrite dictionaries, as in example

    """
    ranges=[]
    for v in bounds.values():
        ranges.append(v)

    # List to store all the combinations
    overwrite_dict_list = []
    for element in product(*ranges):
        d = {}
        for i,key in enumerate(bounds.keys()):
            d[key] = element[i]
        overwrite_dict_list.append(d)
    return overwrite_dict_list

def is_flat_list(l):
    if type(l) != type([]):
        return False

    for x in l:
        if hasattr(x, '__iter__'):
            return False

    return True

def nested_dict_read(keys,dictionary):
    if type(keys) == type(''):
        keys = keys.split(':')
    val = dictionary
    for key in keys:
        if type(val) == type({}):
            val = val[key]
    return val

def nested_dict_write(keys,dictionary,newval):
    t = type(dictionary)
    val = dictionary
    _keys = keys.split(':')
    for key in _keys[:-1]:
        if type(val) == t:
            val = val[key]
    if _keys[-1] in val:
        val[_keys[-1]] = newval
    else:
        print 'Config doesn\'t have: ',keys
        exit()


def hash_anything(d):
    """

    :param d:  any python object should work
    :return: a hash
    """
    h = 0
    if d.__hash__ != None:
        h = hash(d)

    # hash dictionary entries naively
    elif hasattr(d, 'keys'):
        # hash the keys
        h = hash(frozenset(d))

        # hash the values, ignoring order
        for key in d.keys():
            h = h ^ hash_anything(d[key])

    # convert lists to tuples as long as they don't contain iterable objects
    elif is_flat_list(d):
        h = hash_anything(tuple(d))
    else:
        for x in d:
            h = h ^ hash_anything(x)

    return h

def get_n_idx_biggest(arr,n):
    return arr.argsort()[-n:][::-1]

def get_n_idx_smallest(arr,n):
    return arr.argsort()[:n][::-1]

def get_n_idx_near_mean(arr,n):
    mean = arr.mean()
    diff = np.abs(arr - mean)
    return get_n_idx_smallest(diff,n)
