import gzip
import os
import time
from io import BytesIO
from urllib.parse import urlencode

import requests
from astropy.table import Table

from .core import DownloadableBase, FastCsvTable, makedirs_if_needed


class HyperledaQuery(FastCsvTable):
    _read_default_kwargs = dict(format="ascii.fast_csv", comment="#")
    _default_fields = "".join(
        """pgc,objname,objtype,al1950,de1950,al2000,de2000,l2,b2,sgl,sgb,f_astrom,
    type,bar,ring,multiple,compactness,t,e_t,agnclass,logd25,e_logd25,logr25,
    e_logr25,pa,brief,e_brief,ut,e_ut,bt,e_bt,vt,e_vt,it,e_it,kt,e_kt,m21,
    e_m21,mfir,ube,bve,vmaxg,e_vmaxg,vmaxs,e_vmaxs,vdis,e_vdis,vrad,e_vrad,
    vopt,e_vopt,v,e_v,ag,ai,incl,a21,logdc,btc,itc,ubtc,bvtc,bri25,vrot,
    e_vrot,mg2,e_mg2,m21c,hic,vlg,vgsr,vvir,v3k,modz,e_modz,mod0,e_mod0,
    modbest,e_modbest,mabs,e_mabs""".split()
    )
    _default_conditions = ""

    def __init__(self, conditions=None, fields=None, **kwargs):
        path = "http://leda.univ-lyon1.fr/fG.cgi?n=meandata&c=o&of=1,leda,simbad&nra=l&nakd=1&" + urlencode(
            {
                "d": self._default_fields if fields is None else fields,
                "sql": self._default_conditions if conditions is None else conditions,
                "ob": "pgc",
                "a": "csv[,]",
            }
        )
        super().__init__(path, **kwargs)

    def write(self, table, **kwargs):
        raise AttributeError("`HyperledaQuery` do not have write method.")


class EddQuery(DownloadableBase):
    def __init__(self, query, **kwargs):
        self.query = query
        self.kwargs = kwargs

    def _get_raw_output(self):
        return requests.post(
            "http://edd.ifa.hawaii.edu/download.php",
            {"queryall": self.query, "delimiter": ","},
            timeout=(120, 3600),
        ).text

    def read(self):
        return Table.read(BytesIO(self._get_raw_output().encode()), format="ascii.fast_csv")

    def download_as_file(self, file_path, overwrite=False, compress=False):
        makedirs_if_needed(file_path)
        if overwrite or not os.path.isfile(file_path):
            file_open = gzip.open if compress else open
            try:
                with file_open(file_path, "w") as f:
                    f.write("# QUERY: {}\n".format(self.query))
                    f.write("# TIME:  {}\n".format(time.ctime()))
                    f.write(self._get_raw_output())
            except:  # pylint: disable=bare-except # noqa: E722
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                raise
