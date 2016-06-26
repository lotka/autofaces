# autofaces

pip requirements:
```
numpy
matplotlib
tensorflow
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

To visualize the data run the `results2.ipynb` notebook in the notebooks folder
