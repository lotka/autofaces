#!/bin/bash
source ~/Dropbox/.aliasrc
nvds
python test_set_analysis.py ~/v/data/2016_08_05/both_011 final gpu
python test_set_analysis.py ~/v/data/2016_08_05/both_011 early gpu
python test_set_analysis.py ~/v/data/2016_08_05/both_016 final gpu
python test_set_analysis.py ~/v/data/2016_08_05/both_016 early gpu
python test_set_analysis.py ~/v/data/2016_08_05/both_018 final gpu
python test_set_analysis.py ~/v/data/2016_08_05/both_018 early gpu

