from itertools import product
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_images(images,names=None,cmap='Spectral',interpolation='none',title=None,inverse=False,range=None):

    if names == None:
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
        if range == None:
            plt.imshow(images[i],interpolation='none',cmap=cmap)
        else:
            plt.imshow(images[i], interpolation='none', cmap=cmap,vmin=range[0],vmax=range[1])
        plt.title(names[i])
        plt.colorbar()
    plt.show()

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
