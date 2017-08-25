import os
from astropy.table import Table
from .utils import gzip_compress

__all__ = ['Database']


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


class Database(object):
    def __init__(self, root_dir):
        if not os.path.isdir(root_dir):
            raise ValueError('cannot locate {}'.format(root_dir))

        self._root_dir = root_dir

        self._tables = {
            'spectra_clean': FitsTable(os.path.join(self._root_dir, 'data', 'saga_spectra_clean.fits.gz')),
            'hosts_named': GoogleSheets('1GJYuhqfKeuJr-IyyGF_NDLb_ezL6zBiX2aeZFHHPr_s', 0, include_names=['SAGA', 'NSA', 'NGC']),
            'hosts_flag0': GoogleSheets('1b3k2eyFjHFDtmHce1xi6JKuj3ATOWYduTBFftx5oPp8', 1136984451),
            'objects_to_remove': GoogleSheets('1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo', 1379081675, header_start=1),
            'objects_to_add': GoogleSheets('1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo', 286645731, header_start=1),
        }

    def __getitem__(self, key):
        if key in self._tables:
            return self._tables[key]

        if isinstance(key, tuple) and len(key) == 2 and key[0] == 'base':
            path = os.path.join(self._root_dir, 'base_catalogs', 'base_sql_nsa{}.fits.gz'.format(key[1]))
            if os.path.isfile(path):
                self._tables[key] = FitsTable(path)
                return self._tables[key]

        raise KeyError('cannot find {} in database'.format(key))

    def set_nsa_path(self, path, download=False):
        raise NotImplementedError