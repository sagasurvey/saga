import os
from astropy.time import Time
from astropy.table import vstack

from . import read_external
from . import read_observed
from ..utils import add_skycoord

__all__ = ['SpectraData']

class SpectraData(object):
    def __init__(self, spectra_dir_path=None, external_specs_dict=None):
        self.spectra_dir_path = spectra_dir_path
        self._external_specs_dict = external_specs_dict or {}

    def read(self, add_coord=True, before_time=None):
        all_specs = []

        all_specs.extend((getattr(read_external, 'read_'+k.lower())(v) \
                for k, v in self._external_specs_dict.items()))

        if self.spectra_dir_path is not None:
            if before_time is not None and not isinstance(before_time, Time):
                before_time = Time(before_time)
            all_specs.extend([
                read_observed.read_mmt(os.path.join(self.spectra_dir_path, 'MMT'), before_time=before_time),
                read_observed.read_aat(os.path.join(self.spectra_dir_path, 'AAT'), before_time=before_time),
                read_observed.read_aat_mz(os.path.join(self.spectra_dir_path, 'AAT'), before_time=before_time),
                read_observed.read_wiyn(os.path.join(self.spectra_dir_path, 'WIYN')),
                read_observed.read_imacs(os.path.join(self.spectra_dir_path, 'IMACS')),
            ])

        all_specs.extend([
            read_observed.read_deimos(),
            read_observed.read_palomar(),
        ])

        all_specs = vstack(all_specs, 'exact', 'error')

        return add_skycoord(all_specs) if add_coord else all_specs
