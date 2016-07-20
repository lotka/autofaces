import etc
import time

N = 1000
for i, pi in etc.range(N, info_frequency=1):
    time.sleep(0.1)
    if pi:
        print i