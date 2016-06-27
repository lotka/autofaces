# autofaces

general requirements:
```
python 2.7.6
tensorflow
```

pip requirements:
```
numpy
matplotlib
tqdm
sklearn
yaml
ruamel.yaml
```

## To run

``` cd src/ ```


Edit `config/cnn.yaml` to set up experiment parameters


Run an experiment:

```python main.py data_save_path``` (gives you *path_to_results* at the end)

Run the analysis on the test set:

``` python test_set_analysis.py path_to_results model```   *model*=final or early

To visualize the results run the `results.ipynb` notebook in the notebooks folder
