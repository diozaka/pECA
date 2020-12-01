# pECA
An implementation of our pECA approach (Peak Event Coincidence Analysis) to discover statistical relationships between event series and peaks in time series. The methodology is described and analysed in detail in the paper:

> Erik Scharwächter, Emmanuel Müller: **Does Terrorism Trigger Online Hate Speech? On the Association of Events and Time Series.**
> In: Annals of Applied Statistics, Vol. 14, No. 3, 1285-1303, 2020. [[permalink]](http://dx.doi.org/10.1214/20-AOAS1338) [[arXiv preprint]](https://arxiv.org/abs/2004.14733)

We provide a simple Python module ([peca.py](./peca.py)) that can be used to apply our methodology on arbitrary datasets. We also provide two Jupyter notebooks and some required data files to repreat the anaylses and simulations from our paper ([hatespeech.ipynb](./demos/hatespeech.ipynb) and [simulations.ipynb](./demos/simulations.ipynb)). Please note that due to the randomness in the Monte Carlo simulations, the reported Monte Carlo p-values differ slightly from run to run.

## Contact and Citation

* Corresponding author: [Erik Scharwächter](mailto:erik.scharwaechter@cs.tu-dortmund.de)
* Please cite our paper if you use or modify our code for your own work. Here's a `bibtex` snippet:

```
@article{Scharwachter2020,
   title = {{Does terrorism trigger online hate speech? On the association of events and time series}},
   author = {Scharw{\"{a}}chter, Erik and M{\"{u}}ller, Emmanuel},
   journal = {Annals of Applied Statistics},
   number = {3},
   pages = {1285--1303},
   volume = {14},
   year = {2020}
}
```

## Installation and Usage

The `peca` module itself requires `numba`, `numpy` and `scipy`. You can directly install the standalone module from GitHub with pip3:

```bash
$ pip3 install git+https://github.com/diozaka/pECA
```

After installation, please check the module documentation for usage instructions:

```python
>>> import peca
>>> help(peca)
```

The Jupyter notebooks to reproduce our results additionally require `statsmodels`, `matplotlib`, `pandas` and `tqdm`. 

## License

The source codes are released under the [MIT license](./LICENSE). The data in [twitter-volume.csv](./demos/data/twitter-volume.csv) was obtained via the ForSight platform from Crimson Hexagon ([Brandwatch](https://www.brandwatch.com/)).

