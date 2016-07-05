#!/usr/bin/python
import os
import sys
import shutil
from os.path import join

def prefix(i,zeros):
    s = str(i)
    while(len(s) < zeros):
        s = '0' + s
    return s

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(join(a_dir, name))]

top = sys.argv[1]
for dates in get_immediate_subdirectories(top):
    sub = join(top,dates)
    runs = get_immediate_subdirectories(sub)
    for r in runs:
        sub_sub = join(sub,r)
        lock = join(sub_sub,'NOT_FINISHED')
        if os.path.isfile(lock):
            print 'DELETING ',sub_sub
            shutil.rmtree(sub_sub)

    # runs = get_immediate_subdirectories(sub)
    # numbers = []
    # for r in runs:
    #     try:
    #         numbers.append(int(r))
    #     except ValueError:
    #         print 'Could not parse folders ', join(sub,r)
    # if numbers != []:
    #     for i in xrange(1,max(numbers)+1):
    #         d =  join(sub,prefix(i,3))
    #         if not os.path.isdir(d) and (i-1 < len(runs)):
    #             print 'Moving',runs[i-1],'to', prefix(i,3)
    #             shutil.move(join(sub,runs[i-1]), d)

print 'Tidy up complete :)'
