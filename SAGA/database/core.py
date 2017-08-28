import os
import numpy as np
from astropy.table import Table
from astropy.io import fits
from ..utils import gzip_compress


__all__ = ['Database', 'CSVTable', 'GoogleSheets', 'FitsTable', 'DataObject']


class DataObject(object):
    _table = None
    _keep_table_default = False

    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def _read(self):
        raise NotImplementedError

    def _write(self, table):
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
        if overwrite or not os.path.isfile(self.path):
            self._write(table)

    def clear(self):
        self._table = None


class CSVTable(DataObject):
    _keep_table_default = True

    def _read(self):
        return Table.read(self.path, format='ascii.csv', **self.kwargs)

    def _write(self, table):
        table.write(self.path, format='ascii.csv')


class GoogleSheets(DataObject):
    _keep_table_default = True

    def __init__(self, key, gid, **kwargs):
        self.url = 'https://docs.google.com/spreadsheets/d/{0}/export?format=csv&gid={1}'.format(key, gid)
        self.online_url = 'https://docs.google.com/spreadsheets/d/{0}/edit#gid={1}'.format(key, gid)
        self.path = None
        self.kwargs = kwargs

    def _read(self):
        try:
            t = Table.read(self.url, format='ascii.csv', **self.kwargs)
        except FileNotFoundError as e:
            if self.path is not None:
                t = Table.read(self.path, format='ascii.csv', **self.kwargs)
                print("Failed to download table, but loaded local file", self.path)
            else:
                raise e
        return t


class FitsTable(DataObject):
    def __init__(self, path, compress_after_write=True, masked_table=False):
        self.path = path
        self.compress_after_write = compress_after_write
        self.masked_table = masked_table

    def _read(self):
        # From Erik (eteq)
        # the read method *could* work in all cases, but it's slower than the
        # other because it  accounts for masked values... but the SDSS catalog
        # files seem to just be *wrong* in their treatment of masked/null
        # values...so we use this other approach because it doesn't
        # have the masking-associated overhead (which is useless here).
        if self.masked_table:
            return Table.read(self.path, format='fits')
        else:
            with fits.open(self.path) as f:
                return Table(f[1].data)

    def _write(self, table):
        tmp_path = self.path + ('_tmp.fits' if self.compress_after_write else '')
        table.write(tmp_path, format='fits', overwrite=True)
        if self.compress_after_write:
            gzip_compress(tmp_path, self.path)


class NumpyBinary(DataObject):
    def _read(self):
        return np.load(self.path)

    def _write(self, table):
        np.savez(self.path, **table)


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
    >>> saga_database['base', 32].path = 'base_catalogs/base_sql_nsa32.fits.gz'
    >>> saga_database['spectra_clean'].path = 'saga_spectra_clean.fits.gz'

    If you don't have internet, you can do things like
    >>> saga_database['hosts_named'].path = 'hosts_named.csv'
    >>> saga_database['hosts_no_flags'].path = 'hosts_no_flags.csv'
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
            'satellites_named': GoogleSheets('1GJYuhqfKeuJr-IyyGF_NDLb_ezL6zBiX2aeZFHHPr_s', 1),
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
