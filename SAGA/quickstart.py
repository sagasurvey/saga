import os

from .database import Database
from .hosts import FieldCatalog, HostCatalog, HostCuts
from .objects import ObjectCatalog, ObjectCuts
from .targets import TargetSelection

__all__ = ["QuickStart"]


class QuickStart:
    def __init__(self, shared_dir=None, local_dir=None, lowz=False):

        self._shared_dir = (
            shared_dir
            or globals().get("SAGA_DROPBOX")
            or globals().get("saga_dropbox")
            or os.getenv("SAGA_DROPBOX")
            or os.getenv("saga_dropbox")
            or os.getcwd()
        )

        print("SAGA `shared_dir` set to", self._shared_dir)

        self._local_dir = (
            local_dir
            or globals().get("SAGA_DIR")
            or globals().get("saga_dir")
            or os.getenv("SAGA_DIR")
            or os.getenv("saga_dir")
            or os.getcwd()
        )

        print("SAGA `local_dir`  set to", self._local_dir)

        self._lowz = bool(lowz)
        self._database = None
        self._host_catalog = None
        self._object_catalog = None
        self._target_selection = None
        self._host_catalog_class = FieldCatalog if self._lowz else HostCatalog
        self._host_catalog_options = dict()
        self._target_selection_options = dict()

    def set_host_catalog_options(self, **kwargs):
        if kwargs:
            self._host_catalog_options.update(kwargs)
            self._host_catalog = self._object_catalog = self._target_selection = None

    def clear_host_catalog_options(self):
        if self._host_catalog_options:
            self._host_catalog_options = dict()
            self._host_catalog = self._object_catalog = self._target_selection = None

    def set_target_selection_options(self, **kwargs):
        if kwargs:
            self._target_selection_options.update(kwargs)
            self._target_selection = None

    def clear_target_selection_options(self):
        if self._target_selection_options:
            self._target_selection_options = dict()
            self._target_selection = None

    @property
    def database(self):
        if self._database is None:
            self._database = Database(self._shared_dir, self._local_dir)
            if self._lowz:
                self._database.set_default_base_version('v2')
        return self._database

    @property
    def host_catalog(self):
        if self._host_catalog is None:
            self._host_catalog = self._host_catalog_class(
                self.database, **self._host_catalog_options
            )
        return self._host_catalog

    @property
    def object_catalog(self):
        if self._object_catalog is None:
            self._object_catalog = ObjectCatalog(
                self.database, self._host_catalog_class, self.host_catalog
            )
        return self._object_catalog

    @property
    def target_selection(self):
        if self._target_selection is None:
            self._target_selection = TargetSelection(
                self.database,
                self._host_catalog_class,
                host_catalog_instance=self.host_catalog,
                object_catalog_instance=self.object_catalog,
                **self._target_selection_options,
            )
        return self._target_selection

    def __getattr__(self, name):
        if hasattr(ObjectCuts, name):
            return getattr(ObjectCuts, name)
        if hasattr(HostCuts, name):
            return getattr(HostCuts, name)
        raise AttributeError
