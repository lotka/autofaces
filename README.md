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

For GPU support CUDA is required, follow the TensorFlow CUDA set up guide.

## Config Files
`src/config` contains various configuration files for different set ups,
the following examples use `test.yaml`

## To Run

``` cd src/ ```


Run an experiment:

```python main.py --device=cpu --config=config/test.yaml```

This will run the test set analysis, to run it agian:

``` python test_set_analysis.py path_to_results model```   *model*=final or early

To visualize the results run the `viewResults.ipynb` notebook in the notebooks folder,
to compare multiple runs use `compareResults.ipynb`.

To run multiple experiments:

First edit relevant section in main.py under the line:
```if args.compare == True and args.compare != None:```
Then run:

```python main.py --device=cpu --config=config/test.yaml --compare=True```

or to run just one of the generated experiments:

```python main.py --device=cpu --config=config/test.yaml --compare=True --batch=N```
