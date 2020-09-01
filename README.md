SAGA
====
[![arXiv:2008.12783](https://img.shields.io/badge/astro--ph.GA-arXiv%3A2008.12783-B31B1B.svg)](https://arxiv.org/abs/2008.12783)

The [SAGA Survey](http://sagasurvey.org/) is a spectroscopic survey that aims
to determine dwarf galaxy satellite systems around 100 Milky Way analogs down
to the brightness of the Leo I galaxy (Mr < −12.3).

This `SAGA` package contains code to generate and access SAGA data catalogs.
This package is intended for SAGA internal use, but it is licensed under the 
MIT license. 
If you use this package or part of it in your research, please considering citing
SAGA Stage II paper (Mao et al. 2020).

- Visit [yymao/FoFCatalogMatching](https://github.com/yymao/FoFCatalogMatching) 
  if you are looking for the friends-of-friends sky catalog matching code.
- Visit [yymao.github.io/decals-image-list-tool/](https://yymao.github.io/decals-image-list-tool/) 
  if you are looking for the web app for listing cutouts from the egacy Surveys.
- Visit [sagasurvey.org](https://sagasurvey.org) for the most up-to-date SAGA results! 


## Installation

To install the code, run:
```bash
pip install https://github.com/sagasurvey/saga/archive/master.zip
```

To force an update, run
```bash
pip install --upgrade --no-deps --force-reinstall https://github.com/sagasurvey/saga/archive/master.zip
```

The code should be compatible with Python 3.5+,
but has mainly been tested with Python 3.6.

### Dependencies

All required dependencies will be installed automatically.
There are two optional dependencies that require manual installation:

1. casjobs OR sciserver

   You need to install [casjobs](https://github.com/dfm/casjobs) or [sciserver](https://github.com/sciserver/SciScript-Python) to download SDSS catalogs.

   * To install casjobs:
     ```sh
     pip install https://github.com/dfm/casjobs/archive/master.zip
     ```
     (Note: Do *NOT* use `pip install casjobs` as the version on PyPI is outdated.)

   * To install sciserver (recommended):
     ```sh
     pip install "git+https://github.com/sciserver/SciScript-Python.git@sciserver-v2.0.13#egg=sciserver&subdirectory=py3"
     ```
   In both cases you need to set environmental variables to store your credentials. (`CASJOBS_WSID` and `CASJOBS_PW` for casjobs; `SCISERVER_USER` and `SCISERVER_PASS` for sciserver).

2. Extreme Deconvolution

   You only need to install [Extreme Deconvolution](https://github.com/jobovy/extreme-deconvolution) if you want to build GMM (you don't need it to use GMMs).

   ```sh
   pip install https://github.com/jobovy/extreme-deconvolution/archive/master.zip
   ```


## Usage

When using `SAGA`, your code would almost always start with this block:

```python
import SAGA
saga = SAGA.QuickStart('/path/to/saga/dropbox/folder', '/path/to/saga/local/folder')
```

You can then load various datasets:

```python
# load host list
hosts = saga.host_catalog.load()

# load base catalogs of Paper I hosts
base_paper1 = saga.object_catalog.load(hosts='paper1', return_as='list')

# count number of satellites
for base in base_paper1:
    print(base['HOSTID'][0], '# of satellites =', saga.is_sat.count(base))
```

You can find more examples at https://github.com/sagasurvey/examples/tree/master/notebooks

You can also find the schema for the host list and for object catalogs at [SCHEMA.md](SCHEMA.md).
