import os
import sys
import time


def get_times(file_list):
    times = []
    for filename in file_list:
        times.append(os.stat(filename).st_mtime)
    return times

def main():
    print 'watching ', sys.argv[1:]
    prev_times = get_times(sys.argv[1:])
    while True:
        time.sleep(1)
        new_times = get_times(sys.argv[1:])
        if new_times != prev_times:
            break


if __name__ == "__main__":
    main()
