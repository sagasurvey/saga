SAGA
====

This package contains code to access, create and edit SAGA data catalogs.

See the [SAGA Survey website](http://sagasurvey.org/) for details about SAGA.

## Installation

To install the code, run:
```sh
pip install https://github.com/sagasurvey/saga/archive/master.zip
```

To force an update, run
```sh
pip install --upgrade --no-deps --force-reinstall https://github.com/sagasurvey/saga/archive/master.zip
```

The code is designed to be 2-3 compatible, but has mainly been tested with Python 3.6. 


### Dependencies

All required dependencies will be installed automatically. There are two optional dependencies that require manual installation: 

1. casjobs OR sciservevr 
   
   You need to install [casjobs](https://github.com/dfm/casjobs) or [sciservevr](https://github.com/sciserver/SciScript-Python) to download SDSS catalogs. 
   
   * To install casjobs:
     ```sh
     pip install https://github.com/dfm/casjobs/archive/master.zip 
     ```
     (Note: Do *NOT* use `pip install casjobs`)
   
   * To install sciservevr:
     ```sh
     pip install -e "git+https://github.com/sciserver/SciScript-Python.git@sciserver-v2.0.13#egg=SciServer-2.0.13&subdirectory=py3"
     ```
   In both cases you need to set environmental variables to store your credentials. (`CASJOBS_WSID` and `CASJOBS_PW` for casjobs; `SCISERVER_USER` and `SCISERVER_PASS` for sciserver). 
   
2. kcorrect
   
   You need kcorrect to calculate stellar masses. You need to install both the [C code](https://github.com/blanton144/kcorrect) and the [Python wrapper](https://github.com/nirinA/kcorrect_python).
   
   * To install C code:
     1. Obtain the code from https://github.com/blanton144/kcorrect
     2. Follow [this instruction](http://kcorrect.org/#Installing_the_software) and see [some trobuleshooting here](http://kcorrect.org/#Known_problems)
   
   * To install Python wrapper
     1. Make sure you have set the environmental variables `KCORRECT_DIR` and `LD_LIBRARY_PATH` (see [instruction here](https://github.com/nirinA/kcorrect_python#usage))
     2. Run
        ```sh
        pip install https://github.com/nirinA/kcorrect_python/archive/master.zip
        ```

3. Extreme Deconvolution
   
   You only need to install [Extreme Deconvolution](https://github.com/jobovy/extreme-deconvolution) if you want to build GMM (you don't need it to use GMMs). 

   ```sh
   pip install https://github.com/jobovy/extreme-deconvolution/archive/master.zip
   ```


## Example Usage

See more examples at https://github.com/sagasurvey/examples/tree/master/notebooks

```python
import SAGA
from SAGA import ObjectCuts as C

saga_database = SAGA.Database('/path/to/saga/dropbox/folder', '/path/to/saga/local/folder')
saga_host_catalog = SAGA.HostCatalog(saga_database)
saga_object_catalog = SAGA.ObjectCatalog(saga_database)

# load host list (all)
hosts = saga_host_catalog.load('all')

# load host list (no flags, i.e. has SDSS)
hosts_no_flag = saga_host_catalog.load('flag0')

# load all (Paper 1) specs with some basic cuts
specs = saga_object_catalog.load(has_spec=True, cuts=C.basic_cut, version='paper1')

# load base catalogs for all paper1 hosts with the same basic cuts into a list:
base_paper1 = saga_object_catalog.load(hosts='paper1', cuts=C.basic_cut, return_as='list', version='paper1')

# count number of satellites
for base in base_paper1:
    print(base['HOST_NSAID'][0], '# of satellites', C.is_sat.count(base))

# load all base catalogs with the same basic cuts into a list
base_all = saga_object_catalog.load('flag0', cuts=C.basic_cut, return_as='list', version='paper1')
```
