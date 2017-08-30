import os
import shutil
import warnings
import gzip
import numpy as np
import requests
from astropy.table import Table
from astropy.io import fits

try:
    FileExistsError
except NameError:
    FileExistsError = OSError


__all__ = ['Database', 'DataObject', 'FileObject', 'CsvTable', 'GoogleSheets', 'FitsTable', 'NumpyBinary']


class FileObject(object):
    """
    A simple class for file reading (to astropy Table), writing (from astropy Table),
    and download.

    Parameters
    ----------
    path : str
        path or URL to the file

    **kwargs :
        other keyword arguments to pass to astropy.table.Table.read
    """
    def __init__(self, path=None, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def read(self):
        return Table.read(self.path, **self.kwargs)

    def write(self, table):
        return table.write(self.path)

    def download_as_file(self, file_path):
        try:
            r = requests.get(self.path, stream=True)
        except requests.exceptions.MissingSchema:
            shutil.copy(self.path, file_path)
        else:
            r.raw.decode_content = True
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

    def isfile(self):
        if self.path:
            return os.path.isfile(self.path)
        return False


class CsvTable(FileObject):
    def read(self):
        return Table.read(self.path, format='ascii.csv', **self.kwargs)

    def write(self, table):
        return table.write(self.path, format='ascii.csv')


class GoogleSheets(CsvTable):
    def __init__(self, key, gid, **kwargs):
        path = 'https://docs.google.com/spreadsheets/d/{0}/export?format=csv&gid={1}'.format(key, gid)
        self.url = 'https://docs.google.com/spreadsheets/d/{0}/edit#gid={1}'.format(key, gid)
        super(GoogleSheets, self).__init__(path, **kwargs)

    def write(self, table):
        raise NotImplementedError


class FitsTable(FileObject):
    compress_after_write = True

    def read(self):
        with fits.open(self.path, cache=False, lazy_load_hdus=True, **self.kwargs) as f:
            return Table(f[1].data)

    def write(self, table):
        if 'coord' in table.columns and table['coord'].info.dtype.name == 'object':
            table = table.copy()
            del table['coord']
        file_open = gzip.open if self.compress_after_write else open
        with file_open(self.path, 'wb') as f_out:
            table.write(f_out, format='fits')


class NumpyBinary(FileObject):
    def read(self):
        return np.load(self.path, **self.kwargs)

    def write(self, table):
        np.savez(self.path, **table)


class DataObject(object):
    """
    DataObject provide an simple interface to retrive remote data and fall back
    to local copy when necessary.

    Parameters
    ----------
    remote : FileObject or its subclass
        remote FileObject
    local : FileObject or its subclass
        local FileObject
    cache_in_memory : bool, optional
        whether or not to store the table in memory
    use_local_first : bool, optional
        whether or not to try using local file first

    Examples
    --------
    >>> dobj = DataObject(FitsTable('http://somewhere/file.fits'), FitsTable('data/file.fits'))

    You can also do
    >>> dobj = DataObject(FitsTable('http://somewhere/file.fits'))
    >>> dobj.local = 'data/file.fits'
    Or
    >>> dobj.download('data/file.fits')
    """
    def __init__(self, remote, local=None, cache_in_memory=False,
                 use_local_first=False):
        self.remote = remote
        self.local = local
        self.local_type = type(remote) if local is None else type(local)
        self.use_local_first = use_local_first
        self.cache_in_memory = cache_in_memory
        self._cached_table = None


    def _get_local(self):
        if self.local is not None and not isinstance(self.local, self.local_type):
            self.local = self.local_type(self.local)
        return self.local


    def read(self, reload=False):
        """
        Read in the data

        Parameters
        ----------
        reload : bool, optional
            if set to true, ignore cache

        Returns
        -------
        table : astropy.table.Table
        """
        if not reload:
            table = self.retrive_cache()
            if table is not None:
                return table

        if self.use_local_first:
            try:
                table = self._get_local().read()
            except (IOError, OSError):
                warnings.warn("Failed to read local file, try reading the remote file")
                table = self.remote().read()
        else:
            try:
                table = self.remote.read()
            except Exception as read_exception:
                if self._get_local() is None:
                    raise read_exception
                warnings.warn("Failed to read data, fall back to read local file")
                table = self._get_local().read()

        if self.cache_in_memory:
            self.store_cache(table)

        return table


    def write(self, table, dest='remote', overwrite=False):
        """
        write the data to file

        Parameters
        ----------
        table : astropy.table.Table
            data to write
        dest : str
            "remote" or "local"
        overwrite : bool, optional
            if set to true, overwrite existing file
        """
        if dest.lower() == 'remote':
            f = self.remote
        elif dest.lower() == 'local':
            f = self._get_local()
        else:
            raise KeyError('dest must be "remote" or "local"')

        if f.isfile() and not overwrite:
            raise FileExistsError('set overwrite to True to overwrite the file')

        f.write(table)


    def download(self, local_file_path=None, overwrite=False, set_as_local=True):
        """
        Download in the data as a file

        Parameters
        ----------
        local_file_path : str
            local file path
        overwrite : bool, optional
            if set to true, overwrite existing file
        set_as_local : bool, optional
            if set to true (default), use the file at local_file_path as the
            local file for this DataObject
        """
        if local_file_path is None:
            try:
                local_file_path = self._get_local().path
            except AttributeError:
                raise ValueError('Please provide a path for the downloaded file')
            else:
                set_as_local = False # no need to do this again

        if os.path.isfile(local_file_path) and not overwrite:
            warnings.warn('File already exists! Existing file is set as local')
        else:
            self.remote.download_as_file(local_file_path)

        if set_as_local:
            try:
                kwargs = self.remote.kwargs
            except AttributeError:
                kwargs = dict()
            self.local = self.local_type(local_file_path, **kwargs)


    @staticmethod
    def _copy_table(table):
        if table is None:
            return None
        return table.copy()

    def clear_cache(self):
        self._cached_table = None

    def retrive_cache(self):
        return self._copy_table(self._cached_table)

    def store_cache(self, table):
        self._cached_table = self._copy_table(table)


_known_google_sheets = {
    'hosts_named': GoogleSheets('1GJYuhqfKeuJr-IyyGF_NDLb_ezL6zBiX2aeZFHHPr_s', 0, include_names=['SAGA', 'NSA', 'NGC']),
    'hosts_no_flags': GoogleSheets('1b3k2eyFjHFDtmHce1xi6JKuj3ATOWYduTBFftx5oPp8', 1136984451),
    'hosts_no_sdss_flags': GoogleSheets('1b3k2eyFjHFDtmHce1xi6JKuj3ATOWYduTBFftx5oPp8', 1471095077),
    'objects_to_remove': GoogleSheets('1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo', 1379081675, header_start=1),
    'objects_to_add': GoogleSheets('1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo', 286645731, header_start=1),
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
            'spectra_clean': DataObject(FitsTable(os.path.join(self._root_dir, 'data', 'saga_spectra_clean.fits.gz'))),
            'nsa_v1.0.1': DataObject(FitsTable('https://data.sdss.org/sas/dr14/sdss/atlas/v1/nsa_v1_0_1.fits'), use_local_first=True),
            'nsa_v0.1.2': DataObject(FitsTable('http://sdss.physics.nyu.edu/mblanton/v0/nsa_v0_1_2.fits'), use_local_first=True),
            ('spec', 'gama'): DataObject(FitsTable('http://www.gama-survey.org/dr2/data/cat/SpecCat/v08/SpecObj.fits'), use_local_first=True),
        }

        self._tables['nsa'] = self._tables['nsa_v1.0.1']

        for k, v in _known_google_sheets.items():
            self._tables[k] = DataObject(v, CsvTable(), cache_in_memory=True)

        self.base_file_path_pattern = os.path.join(self._root_dir, 'base_catalogs', 'base_sql_nsa{}.fits.gz')
        self.sdss_file_path_pattern = 'sdss_nsa{}.fits.gz'
        self.wise_file_path_pattern = 'wise_nsa{}.fits'

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
