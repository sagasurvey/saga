import os

import numpy as np
from astropy.table import join, vstack
from astropy.time import Time

from ..utils import add_skycoord
from . import read_external, read_observed
from .common import ensure_specs_dtype

__all__ = ["SpectraData"]


class SpectraData(object):
    def __init__(
        self, spectra_dir_path=None, external_specs_dict=None, halpha_data_obj=None
    ):
        self.spectra_dir_path = spectra_dir_path
        self._external_specs_dict = external_specs_dict or {}
        self.halpha_data_obj = halpha_data_obj

    def read(self, add_coord=True, before_time=None, additional_specs=None):
        all_specs = []

        for k, v in self._external_specs_dict.items():
            func_name = "read_" + k.lower()
            func = getattr(read_external, func_name, None) or getattr(
                read_observed, func_name, None
            )
            if func is None:
                print("Cannot find function to read {}".format(k))
                continue
            all_specs.append(func(v))

        if self.spectra_dir_path is not None:
            if before_time is not None and not isinstance(before_time, Time):
                before_time = Time(before_time)
            all_specs.extend(
                [
                    read_observed.read_mmt(
                        os.path.join(self.spectra_dir_path, "MMT"),
                        before_time=before_time,
                    ),
                    read_observed.read_aat(
                        os.path.join(self.spectra_dir_path, "AAT"),
                        before_time=before_time,
                    ),
                    read_observed.read_aat_mz(
                        os.path.join(self.spectra_dir_path, "AAT"),
                        before_time=before_time,
                    ),
                    read_observed.read_aat_mz(
                        os.path.join(self.spectra_dir_path, "AAT_LOWZ"),
                        before_time=before_time,
                    ),
                    read_observed.read_wiyn(
                        os.path.join(self.spectra_dir_path, "WIYN")
                    ),
                    read_observed.read_imacs(
                        os.path.join(self.spectra_dir_path, "IMACS")
                    ),
                ]
            )

        all_specs.append(read_observed.read_deimos())

        if additional_specs:
            all_specs.extend(ensure_specs_dtype(spec) for spec in additional_specs)

        all_specs = vstack([specs for specs in all_specs if specs is not None], "exact")

        if self.halpha_data_obj is not None and self.halpha_data_obj.remote.isfile():
            halpha = self.halpha_data_obj.read()[
                "EW_Halpha", "EW_Halpha_err", "SPECOBJID1", "MASKNAME"
            ]
            halpha.rename_column("SPECOBJID1", "SPECOBJID")
            halpha = ensure_specs_dtype(halpha, skip_missing_cols=True)
            del all_specs["EW_Halpha"], all_specs["EW_Halpha_err"]
            all_specs = join(all_specs, halpha, ["SPECOBJID", "MASKNAME"], "left")
            all_specs["EW_Halpha"].fill_value = np.nan
            all_specs["EW_Halpha_err"].fill_value = np.nan

        all_specs = all_specs.filled()

        return add_skycoord(all_specs) if add_coord else all_specs
