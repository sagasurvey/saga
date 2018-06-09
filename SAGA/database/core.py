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


__all__ = ['DataObject', 'FileObject', 'CsvTable', 'GoogleSheets', 'FitsTable', 'NumpyBinary']


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
        self._makedirs_if_needed(self.path)
        return table.write(self.path)

    @staticmethod
    def _makedirs_if_needed(path):
        dirs, fn = os.path.split(path)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    def download_as_file(self, file_path, overwrite=False, compress=False):
        self._makedirs_if_needed(file_path)
        if overwrite or not os.path.isfile(file_path):
            try:
                r = requests.get(self.path, stream=True)
            except requests.exceptions.MissingSchema:
                shutil.copy(self.path, file_path)
            else:
                r.raw.decode_content = True
                file_open = gzip.open if compress else open
                try:
                    with file_open(file_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                except:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    raise

    def isfile(self):
        if self.path:
            return os.path.isfile(self.path)
        return False


class CsvTable(FileObject):
    def read(self):
        return Table.read(self.path, format='ascii.csv', **self.kwargs)

    def write(self, table):
        self._makedirs_if_needed(self.path)
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
            return Table(f[1].data) # pylint: disable=E1101

    def write(self, table):
        if 'coord' in table.columns and table['coord'].info.dtype.name == 'object':
            table = table.copy()
            del table['coord']
        file_open = gzip.open if self.compress_after_write else open
        self._makedirs_if_needed(self.path)
        with file_open(self.path, 'wb') as f_out:
            table.write(f_out, format='fits')


class NumpyBinary(FileObject):
    def read(self):
        return np.load(self.path, **self.kwargs)

    def write(self, table):
        self._makedirs_if_needed(self.path)
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
                warnings.warn("Failed to read local file, will try reading the remote file")
                table = self.remote.read()
        else:
            try:
                table = self.remote.read()
            except Exception as read_exception: #pylint: disable=W0703
                if self._get_local() is None:
                    raise read_exception
                warnings.warn("Failed to read data, falling back to try to read local file")
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


    def download(self, local_file_path=None, overwrite=False, compress=False, set_as_local=True):
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
                pass
            else:
                set_as_local = False # no need to do this again

        self.remote.download_as_file(local_file_path, overwrite=overwrite, compress=compress)

        if set_as_local:
            kwargs = dict()
            try:
                kwargs = self._get_local().kwargs
            except AttributeError:
                try:
                    kwargs = self.remote.kwargs
                except AttributeError:
                    pass
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
