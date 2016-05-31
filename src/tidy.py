#!/usr/bin/python
import os
import sys
import shutil

def prefix(i,zeros):
    s = str(i)
    while(len(s) < zeros):
        s = '0' + s
    return s

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

top = sys.argv[1]
for dates in get_immediate_subdirectories(top):
    sub = os.path.join(top,dates)
    runs = get_immediate_subdirectories(sub)
    for r in runs:
        sub_sub = os.path.join(sub,r)
        lock = os.path.join(sub_sub,'NOT_FINISHED')
        if os.path.isfile(lock):
            print 'DELETING ',sub_sub
            shutil.rmtree(sub_sub)

    runs = get_immediate_subdirectories(sub)
    numbers = []
    for r in runs:
        numbers.append(int(r))
    if numbers != []:
        for i in xrange(1,max(numbers)+1):
            d =  os.path.join(sub,prefix(i,3))
            if not os.path.isdir(d) and (i-1 < len(runs)):
                print 'Moving',runs[i-1],'to', prefix(i,3)
                shutil.move(os.path.join(sub,runs[i-1]), d)

print 'Tidy up complete :)'
