import matplotlib
matplotlib.use('Agg')
import results
runs = []
runs.append(('2016_08_14','alpha',4))
runs.append(('2016_08_25','alpha', 1))
runs.append(('2016_08_24','alpha', 3))
runs.append(('2016_08_25','alpha', 4))
runs.append(('2016_08_24','alpha', 6))
runs.append(('2016_08_31','transfer', 1))
runs.append(('2016_08_31','transfer', 4))

for run in runs:
    date,group,id = run
    r = results.Results(date,id,local=False,experiment_group=group)
    r.report_losses()
