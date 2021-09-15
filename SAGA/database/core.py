import gzip
import logging
import os
import shutil
from abc import ABC, abstractmethod

import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import clear_download_cache

from ..utils import makedirs_if_needed

try:
    FileExistsError
except NameError:
    FileExistsError = OSError  # pylint: disable=redefined-builtin


__all__ = [
    "DownloadableBase",
    "DataObject",
    "FileObject",
    "CsvTable",
    "FastCsvTable",
    "EcsvTable",
    "GoogleSheets",
    "FitsTableGeneric",
    "FitsTable",
    "NumpyBinary",
]


class DownloadableBase(ABC):
    @abstractmethod
    def download_as_file(self, file_path, overwrite=False, compress=False):
        pass


class FileObject(DownloadableBase):
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

    _read_default_kwargs = dict()
    _write_default_kwargs = dict()

    def __init__(self, path=None, **kwargs):
        self.path = path
        self.kwargs = kwargs

    @staticmethod
    def _gz_fallback(file_path):
        file_path_str = str(file_path)
        file_path_alt = file_path_str[:-3] if file_path_str.lower().endswith(".gz") else (file_path_str + ".gz")
        if (not os.path.isfile(file_path)) and os.path.isfile(file_path_alt):
            return file_path_alt
        return file_path

    def read(self):
        kwargs_this = dict(self._read_default_kwargs, **self.kwargs)
        path = self._gz_fallback(self.path)
        return Table.read(path, **kwargs_this)

    def write(self, table, **kwargs):
        kwargs_this = dict(self._write_default_kwargs, **kwargs)
        makedirs_if_needed(self.path)
        return table.write(self.path, **kwargs_this)

    def download_as_file(self, file_path, overwrite=False, compress=False):
        makedirs_if_needed(file_path)
        if overwrite or not os.path.isfile(file_path):
            try:
                r = requests.get(self.path, stream=True, timeout=(120, 3600))
            except requests.exceptions.MissingSchema:
                shutil.copy(self.path, file_path)
            else:
                file_open = gzip.open if (compress or file_path.endswith(".gz")) else open
                try:
                    with file_open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=(16 * 1024 * 1024)):
                            f.write(chunk)
                except:  # noqa: E722
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    raise
            finally:
                r.close()

    def isfile(self):
        if self.path:
            return os.path.isfile(self._gz_fallback(self.path))
        return False


class CsvTable(FileObject):
    _read_default_kwargs = dict(format="ascii.csv")
    _write_default_kwargs = dict(format="ascii.csv", overwrite=True)


class FastCsvTable(FileObject):
    _read_default_kwargs = dict(format="ascii.fast_csv")
    _write_default_kwargs = dict(format="ascii.fast_csv", overwrite=True)


class EcsvTable(FileObject):
    _read_default_kwargs = dict(format="ascii.ecsv")
    _write_default_kwargs = dict(format="ascii.ecsv", overwrite=True)


class GoogleSheets(FastCsvTable):
    def __init__(self, key, gid, **kwargs):
        path = "https://docs.google.com/spreadsheets/d/{0}/export?format=csv&gid={1}".format(key, gid)
        self.url = "https://docs.google.com/spreadsheets/d/{0}/edit#gid={1}".format(key, gid)
        super(GoogleSheets, self).__init__(path, **kwargs)

    def read(self):
        clear_download_cache(self.path)
        return super(GoogleSheets, self).read()

    def write(self, table, **kwargs):
        raise NotImplementedError


class FitsTableGeneric(FileObject):
    _read_default_kwargs = dict(memmap=True)
    _write_default_kwargs = dict(format="fits", overwrite=True)


class FitsTable(FileObject):
    compress_after_write = False
    _read_default_kwargs = dict(cache=False, memmap=True)
    _write_default_kwargs = dict(format="fits", overwrite=True)

    def read(self):
        kwargs_this = dict(self._read_default_kwargs, **self.kwargs)
        path = self._gz_fallback(self.path)

        try:
            hdu_list = fits.open(path, **kwargs_this)
        except OSError:
            # this helps fits.open guess the compression better
            hdu_list = fits.open(open(path, "rb"), **kwargs_this)

        try:
            t = Table(hdu_list[1].data, masked=False)
        finally:
            try:
                del hdu_list[1].data  # pylint: disable=no-member
                hdu_list.close()
                del hdu_list
            except:  # pylint: disable=bare-except  # noqa: E722
                pass

        return t

    def write(self, table, **kwargs):
        coord = None
        if "coord" in table.columns and table["coord"].info.dtype.name == "object":
            coord = table["coord"]
            del table["coord"]
        compress = self.compress_after_write or self.path.endswith(".gz")
        file_open = gzip.open if compress else open
        makedirs_if_needed(self.path)
        kwargs_this = dict(self._write_default_kwargs, **kwargs)
        with file_open(self.path, "wb") as f_out:
            table.write(f_out, **kwargs_this)
        if coord is not None:
            table["coord"] = coord


class NumpyBinary(FileObject):
    def read(self):
        path = self._gz_fallback(self.path)
        return np.load(path, **self.kwargs)

    def write(self, table, **kwargs):
        makedirs_if_needed(self.path)
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

    def __init__(self, remote, local=None, cache_in_memory=False, use_local_first=False):
        self._local = None
        self.remote = remote
        self.local = local
        self.use_local_first = bool(use_local_first)
        self.cache_in_memory = bool(cache_in_memory)
        self._cached_table = None

        if use_local_first and local is None:
            raise ValueError("Must specify `local` when setting `use_local_first=True`.")

    @property
    def local(self):
        return self._local

    @local.setter
    def local(self, value):
        if value is None:
            self._local = None
        elif isinstance(value, FileObject):
            self._local = value
        elif isinstance(self._local, FileObject):
            self._local = type(self._local)(value, **self._local.kwargs)
        elif isinstance(self.remote, FileObject):
            self._local = type(self.remote)(value, **self.remote.kwargs)
        else:
            self._local = FileObject(value)

    def read(self, reload=False, **kwargs):
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
            if not self.local.isfile():
                logging.warning("Cannot find local file; attempt to download from remote...")
                self.download()
            try:
                table = self.local.read()
            except (IOError, OSError):
                logging.warning("Failed to read local file; attempt to read remote file...")
                table = self.remote.read(**kwargs)
        else:
            try:
                table = self.remote.read(**kwargs)
            except Exception as read_exception:  # pylint: disable=W0703
                if self.local is None:
                    raise read_exception
                logging.warning("Failed to read remote; fall back to read local file...")
                if not self.local.isfile():
                    logging.warning("Cannot find local file; attempt to download from remote...")
                    self.download()
                table = self.local.read()

        if self.cache_in_memory:
            self.store_cache(table)

        return table

    def write(self, table, dest=None, overwrite=False):
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
        if dest is None:
            dest = "local" if self.use_local_first else "remote"
        if dest.lower() == "remote":
            f = self.remote
        elif dest.lower() == "local":
            f = self.local
        else:
            raise KeyError('dest must be "remote" or "local"')

        if f.isfile() and not overwrite:
            raise FileExistsError("set overwrite to True to overwrite the file")

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
                local_file_path = self.local.path
            except AttributeError:
                pass
            else:
                set_as_local = False  # no need to do this again

        self.remote.download_as_file(local_file_path, overwrite=overwrite, compress=compress)

        if set_as_local:
            self.local = local_file_path

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

    @property
    def path(self):
        return self.local.path if self.use_local_first else self.remote.path

    def isfile(self):
        return self.local.isfile() if self.use_local_first else self.remote.isfile()
