import os
import numpy as np
from astropy.table import Table
from ..utils import gzip_compress

class DataObject(object):
    _table = None
    _keep_table_default = False

    def _read(self):
        raise NotImplementedError

    def _write(self, table, overwrite):
        raise NotImplementedError

    def read(self, reload=False, keep=None):
        if reload or self._table is None:
            table = self._read()
            if keep is None:
                keep = self._keep_table_default
            if keep:
                self._table = table
        else:
            table = self._table
        return table

    def write(self, table, overwrite=False):
        self._write(table, overwrite)

    def clear(self):
        self._table = None


class GoogleSheets(DataObject):
    _keep_table_default = True

    def __init__(self, key, gid, **kwargs):
        self._url = 'https://docs.google.com/spreadsheets/d/{0}/export?format=csv&gid={1}'.format(key, gid)
        self._kwargs = kwargs

    def _read(self):
        return Table.read(self._url, format='ascii.csv', **self._kwargs)


class FitsTable(DataObject):
    def __init__(self, path, compress_after_write=True):
        self._path = path
        self._compress_after_write = compress_after_write

    def _read(self):
        return Table.read(self._path, format='fits')

    def _write(self, table, overwrite=False):
        if overwrite or not os.path.isfile(self._path):
            tmp_path = self._path + ('_tmp.fits' if self._compress_after_write else '')
            table.write(tmp_path, format='fits', overwrite=True)
            if self._compress_after_write:
                gzip_compress(tmp_path, self._path)


class NumpyBinary(DataObject):
    def __init__(self, path):
        self._path = path

    def _read(self):
        return np.load(self._path)

    def _write(self, table, overwrite=False):
        if overwrite or not os.path.isfile(self._path):
            np.savez(self._path, **table)


class Database(object):
    """
    This class provide the interface between the filesystem and other parts of
    the SAGA package.

    Parameters
    ----------
    root_dir : str, optional
        path to the shared SAGA Dropbox root directory
        if you don't have access, set to None.

    Examples
    --------
    >>> import SAGA
    >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox')
    >>> saga_hosts = SAGA.HostCatalog(saga_database)
    >>> saga_objects = SAGA.ObjectCatalog(saga_database)


    If you don't have access to SAGA Dropbox, you can do:

    >>> saga_database = SAGA.Database()
    >>> saga_database.set_base_fits_file_path('base_catalogs/base_sql_nsa32.fits.gz', 32)
    >>> saga_database.set_spectra_clean_fits_file_path('saga_spectra_clean.fits.gz')
    >>> saga_hosts = SAGA.HostCatalog(saga_database)
    >>> saga_objects = SAGA.ObjectCatalog(saga_database)

    """
    def __init__(self, root_dir=None):
        if root_dir is not None and not os.path.isdir(root_dir):
            raise ValueError('cannot locate {}'.format(root_dir))

        self._root_dir = root_dir

        self._tables = {
            'hosts_named': GoogleSheets('1GJYuhqfKeuJr-IyyGF_NDLb_ezL6zBiX2aeZFHHPr_s', 0, include_names=['SAGA', 'NSA', 'NGC']),
            'hosts_no_flags': GoogleSheets('1b3k2eyFjHFDtmHce1xi6JKuj3ATOWYduTBFftx5oPp8', 1136984451),
            'hosts_no_sdss_flags': GoogleSheets('1b3k2eyFjHFDtmHce1xi6JKuj3ATOWYduTBFftx5oPp8', 1471095077),
            'objects_to_remove': GoogleSheets('1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo', 1379081675, header_start=1),
            'objects_to_add': GoogleSheets('1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo', 286645731, header_start=1),
        }

        if self._root_dir is not None:
            self._tables['gmm_parameters'] = NumpyBinary(os.path.join(self._root_dir, 'data', 'gmm_parameters.npz'))
            self._tables['spectra_clean'] = FitsTable(os.path.join(self._root_dir, 'data', 'saga_spectra_clean.fits.gz'))

    def __getitem__(self, key):
        if key in self._tables:
            return self._tables[key]

        if isinstance(key, tuple) and len(key) == 2 and key[0] == 'base':
            path = os.path.join(self._root_dir, 'base_catalogs', 'base_sql_nsa{}.fits.gz'.format(key[1]))
            if os.path.isfile(path):
                self._tables[key] = FitsTable(path)
                return self._tables[key]

        raise KeyError('cannot find {} in database'.format(key))

    def set_base_fits_file_path(self, host_nsa_id, path):
        """
        this function should not be used, but just in case you don't
        have access to the SAGA dropbox but a single fits file for one host

        Parameters
        ----------
        host_nsa_id : int
            host nsa id

        path : str:
            path to the fits (or fits.gz) file
        """
        if os.path.isfile(path):
            self._tables[('base', int(host_nsa_id))] = FitsTable(path)

    def set_spectra_clean_fits_file_path(self, path):
        """
        this function should not be used, but just in case you don't
        have access to the SAGA dropbox but a single fits file for spectra

        Parameters
        ----------
        path : str:
            path to the fits (or fits.gz) file
        """
        if os.path.isfile(path):
            self._tables['spectra_clean'] = FitsTable(path)

