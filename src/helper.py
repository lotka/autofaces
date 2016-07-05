def is_flat_list(l):
    if type(l) != type([]):
        return False

    for x in l:
        if hasattr(x, '__iter__'):
            return False

    return True


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