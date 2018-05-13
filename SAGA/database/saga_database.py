import os
from .core import DataObject, CsvTable, GoogleSheets, FitsTable, NumpyBinary
from .spectra import SpectraData

__all__ = ['known_google_sheets', 'Database']

known_google_sheets = {
    'hosts_named': GoogleSheets('1GJYuhqfKeuJr-IyyGF_NDLb_ezL6zBiX2aeZFHHPr_s', 0, include_names=['SAGA', 'NSA', 'NGC']),
    'hosts_no_flags': GoogleSheets('1b3k2eyFjHFDtmHce1xi6JKuj3ATOWYduTBFftx5oPp8', 1136984451),
    'hosts_no_sdss_flags': GoogleSheets('1b3k2eyFjHFDtmHce1xi6JKuj3ATOWYduTBFftx5oPp8', 1471095077),
    'objects_to_remove': GoogleSheets('1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo', 1379081675, header_start=1, include_names=['SDSS ID', 'Targ_RA', 'Targ_Dec']),
    'objects_to_add': GoogleSheets('1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo', 286645731, header_start=1, include_names=['SDSS ID', 'Targ_RA', 'Targ_Dec']),
    'satellites_named': GoogleSheets('1GJYuhqfKeuJr-IyyGF_NDLb_ezL6zBiX2aeZFHHPr_s', 1),
}

class Database(object):
    """
    This class provide the interface between the filesystem and other parts of
    the SAGA package.

    Parameters
    ----------
    root_dir : str, optional
        path to the shared SAGA Dropbox root directory
        if you don't have access, set to None.

    Notes
    -----
    Use database[key].read() to access the file.
    Use database[key].path to overwrite the file path.
    For base catalogs, key should be in the form of ('base', host_id).

    Examples
    --------
    >>> import SAGA
    >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox')
    >>> saga_hosts = SAGA.HostCatalog(saga_database)
    >>> saga_objects = SAGA.ObjectCatalog(saga_database)

    If you don't have access to SAGA Dropbox, you can do:
    >>> saga_database = SAGA.Database()
    >>> saga_database['base', 32].local = 'base_catalogs/base_sql_nsa32.fits.gz'
    >>> saga_database['spectra_clean'].local = 'data/saga_spectra_clean.fits.gz'

    If you don't have internet, you can do things like
    >>> saga_database['hosts_named'].local = 'hosts_named.csv'
    >>> saga_database['hosts_no_flags'].local = 'hosts_no_flags.csv'

    If you have internet now, but want to download the file just in case:
    >>> saga_database['hosts_named'].download('hosts_named.csv')
    >>> saga_database['hosts_no_flags'].download('hosts_no_flags.csv')
    """

    def __init__(self, root_dir=None):
        if root_dir is None:
            root_dir = os.curdir

        if not os.path.isdir(root_dir):
            raise ValueError('cannot locate {}'.format(root_dir))

        self._root_dir = root_dir

        self._tables = {
            'gmm_parameters': DataObject(NumpyBinary(os.path.join(self._root_dir, 'data', 'gmm_parameters.npz'))),
            'gmm_parameters_decals': DataObject(NumpyBinary(os.path.join(self._root_dir, 'data', 'gmm_parameters_decals.npz'))),
            'gmm_parameters_no_outlier': DataObject(NumpyBinary(os.path.join(self._root_dir, 'data', 'gmm_parameters_no_outlier.npz'))),
            'gmm_parameters_201708': DataObject(NumpyBinary(os.path.join(self._root_dir, 'data', 'gmm_parameters_201708.npz'))),
            'spectra_clean': DataObject(FitsTable(os.path.join(self._root_dir, 'data', 'saga_spectra_clean.fits.gz'))),
            'nsa_v1.0.1': DataObject(FitsTable('https://data.sdss.org/sas/dr14/sdss/atlas/v1/nsa_v1_0_1.fits'), use_local_first=True),
            'nsa_v0.1.2': DataObject(FitsTable('http://sdss.physics.nyu.edu/mblanton/v0/nsa_v0_1_2.fits'), use_local_first=True),
            'spectra_gama_dr2': DataObject(FitsTable('http://www.gama-survey.org/dr2/data/cat/SpecCat/v08/SpecObj.fits'), use_local_first=True),
            'spectra_gama': DataObject(FitsTable('http://www.gama-survey.org/dr3/data/cat/SpecCat/v27/SpecObj.fits'), use_local_first=True),
        }

        self._tables['nsa'] = self._tables['nsa_v0.1.2']
        self._tables['spectra_raw_all'] = DataObject(SpectraData(os.path.join(self._root_dir, 'Spectra', 'Final'), self._tables['spectra_gama']), FitsTable())

        for k, v in known_google_sheets.items():
            self._tables[k] = DataObject(v, CsvTable(), cache_in_memory=True)

        self._file_path_pattern = {'base': os.path.join(self._root_dir, 'base_catalogs', 'base_sql_nsa{}.fits.gz')}


    def _set_file_path_pattern(self, key, value):
        self._file_path_pattern[key] = value
        keys_to_del = [k for k in self._tables if isinstance(k, tuple) and len(k) == 2 and k[0] == key]
        for k in keys_to_del:
            del self._tables[k]


    @property
    def base_file_path_pattern(self):
        return self._file_path_pattern['base']

    @base_file_path_pattern.setter
    def base_file_path_pattern(self, value):
        self._set_file_path_pattern('base', value)

    @property
    def sdss_file_path_pattern(self):
        return self._file_path_pattern['sdss']

    @sdss_file_path_pattern.setter
    def sdss_file_path_pattern(self, value):
        self._set_file_path_pattern('sdss', value)

    @property
    def wise_file_path_pattern(self):
        return self._file_path_pattern['wise']

    @wise_file_path_pattern.setter
    def wise_file_path_pattern(self, value):
        self._set_file_path_pattern('wise', value)

    @property
    def des_file_path_pattern(self):
        return self._file_path_pattern['des']

    @des_file_path_pattern.setter
    def des_file_path_pattern(self, value):
        self._set_file_path_pattern('des', value)

    @property
    def decals_file_path_pattern(self):
        return self._file_path_pattern['decals']

    @decals_file_path_pattern.setter
    def decals_file_path_pattern(self, value):
        self._set_file_path_pattern('decals', value)

    def __getitem__(self, key):
        if key in self._tables:
            return self._tables[key]

        if isinstance(key, tuple) and len(key) == 2 and key[0] in ('base', 'sdss', 'wise'):
            path = getattr(self, '{}_file_path_pattern'.format(key[0])).format(key[1])
            self._tables[key] = DataObject(FitsTable(path))
            return self._tables[key]

        raise KeyError('cannot find {} in database'.format(key))


    def __setitem__(self, key, value):
        self._tables[key] = value


    def __delitem__(self, key):
        del self._tables[key]


    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


    def keys(self):
        return self._tables.keys()
