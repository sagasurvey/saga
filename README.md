SAGA
====

This package contains code to access, create and edit SAGA data catalogs.

See the [SAGA Survey website](http://sagasurvey.org/) for details about SAGA.

## Installation

    pip install pip install git+git://github.com/sagasurvey/saga.git


## Example Usage

```python
import SAGA
import SAGA.objects.queries as Q

saga_database = SAGA.Database('/path/to/SAGA/data/folder')
saga_hosts = SAGA.Hosts(saga_database)
saga_objects = SAGA.Objects(saga_database)

# load host list
hosts = saga_hosts.load()

# load all specs with some basic cuts
basic_query = Q.is_clean & Q.sat_rcut & Q.is_galaxy & Q.fibermag_r_cut & Q.faint_end_limit
specs = saga_objects.load(has_spec=True, query=basic_query)

# load all base catalogs with the same basic cuts
base_all = [saga_objects.load(hosts='all', query=basic_query, iter_hosts=True)]
```
