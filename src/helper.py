from itertools import product

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
    val = dictionary
    for key in keys[:-1]:
        if type(val) == type({}):
            val = val[key]
    val[keys[-1]] = newval


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