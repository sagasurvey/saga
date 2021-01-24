import os

from astropy.table import vstack
from astropy.time import Time

from ..utils import add_skycoord
from . import read_external, read_observed
from .common import ensure_specs_dtype

__all__ = ["SpectraData"]


class SpectraData(object):
    def __init__(self, spectra_dir_path=None, external_specs_dict=None):
        self.spectra_dir_path = spectra_dir_path
        self._external_specs_dict = external_specs_dict or {}

    def read(
        self,
        add_coord=True,
        before_time=None,
        additional_specs=None,
        exclude_spec_masks=None,
    ):
        all_specs = []

        for k, v in self._external_specs_dict.items():
            func_name = "read_" + k.lower()
            func = getattr(read_external, func_name, None) or getattr(read_observed, func_name, None)
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
                        exclude_spec_masks=exclude_spec_masks,
                    ),
                    read_observed.read_mmt_bino(
                        os.path.join(self.spectra_dir_path, "MMT_BINO"),
                        exclude_spec_masks=exclude_spec_masks,
                    ),
                    read_observed.read_aat(
                        os.path.join(self.spectra_dir_path, "AAT"),
                        before_time=before_time,
                        exclude_spec_masks=exclude_spec_masks,
                    ),
                    read_observed.read_aat_mz(
                        os.path.join(self.spectra_dir_path, "AAT"),
                        before_time=before_time,
                        exclude_spec_masks=exclude_spec_masks,
                    ),
                    read_observed.read_aat_mz(
                        os.path.join(self.spectra_dir_path, "AAT_LOWZ"),
                        before_time=before_time,
                        exclude_spec_masks=exclude_spec_masks,
                    ),
                    read_observed.read_wiyn(os.path.join(self.spectra_dir_path, "WIYN")),
                    read_observed.read_imacs(os.path.join(self.spectra_dir_path, "IMACS")),
                ]
            )

        all_specs.append(read_observed.read_deimos())

        if additional_specs:
            all_specs.extend(ensure_specs_dtype(spec) for spec in additional_specs)

        all_specs = vstack([specs for specs in all_specs if specs is not None], "exact")

        return add_skycoord(all_specs) if add_coord else all_specs
