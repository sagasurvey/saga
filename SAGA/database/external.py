import gzip
import os
import random
import re
import shutil
import string
import time
import warnings

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table, vstack

try:
    from astroquery.gaia import Gaia
except ImportError:
    _HAS_GAIA_ = False
else:
    _HAS_GAIA_ = True
    Gaia.ROW_LIMIT = 0

from ..utils import makedirs_if_needed
from ..utils.overlap_checker import is_within
from .core import DownloadableBase, FitsTable

_HAS_CASJOBS_ = True
try:
    import casjobs
except ImportError:
    _HAS_CASJOBS_ = False

_HAS_SCISERVER_ = True
try:
    import SciServer.Authentication
    import SciServer.CasJobs
except ImportError:
    _HAS_SCISERVER_ = False

# fix astropy six
try:
    import astropy.extern.six  # noqa: F401
except ImportError:
    import sys

    import six

    sys.modules["astropy.extern.six"] = six

_HAS_DATALAB_ = True
try:
    import dl.queryClient
except ImportError:
    _HAS_DATALAB_ = False


__all__ = [
    "SdssQuery",
    "WiseQuery",
    "DesQuery",
    "DecalsPrebuilt",
    "DecalsQuery",
    "GaiaQuery",
    "download_catalogs_for_hosts",
]


def get_random_string(length=6):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def ensure_deg(value):
    if isinstance(value, u.Quantity):
        return value.to(u.deg).value  # pylint: disable=E1101
    return float(value)


class WiseQuery(FitsTable):
    """
    Examples
    --------
    WiseQuery(ra, dec).download_as_file('/path/to/file', overwrite=False)

    """

    def __init__(self, ra, dec, radius=1.0, **kwargs):
        path = "http://unwise.me/phot_near/?ra={:f}&dec={:f}&radius={:f}&datatype=flat&version=sdss-dr10d".format(
            ensure_deg(ra), ensure_deg(dec), ensure_deg(radius)
        )
        super(WiseQuery, self).__init__(path, **kwargs)

    def write(self, table, **kwargs):
        raise NotImplementedError


class SdssQuery(DownloadableBase):
    """
    Examples
    --------
    >>> SdssQuery(ra, dec, context='DR14').download_as_file('/path/to/file', overwrite=False)

    Notes
    -----
    For SdssQuery to work , please follow these instructions:

    1. Get an account from  http://skyserver.sdss3.org/CasJobs/CreateAccount.aspx

    2. Edit your `.bashrc`:
        export CASJOBS_WSID='2090870927'   # get your WSID from site above
        export CASJOBS_PW='my password'
    """

    _query_template = """
        SELECT  p.objId  as OBJID,
        p.ra as RA, p.dec as DEC,
        p.type as PHOTPTYPE,  dbo.fPhotoTypeN(p.type) as PHOT_SG,

        p.flags as FLAGS, p.clean,
        flags & dbo.fPhotoFlags('SATURATED') as SATURATED,
        flags & dbo.fPhotoFlags('BAD_COUNTS_ERROR') as BAD_COUNTS_ERROR,
        flags & dbo.fPhotoFlags('BINNED1') as BINNED1,
        flags & dbo.fPhotoFlags('TOO_FEW_GOOD_DETECTIONS') as TOO_FEW_GOOD_DETECTIONS,

        p.modelMag_u as u, p.modelMag_g as g, p.modelMag_r as r,p.modelMag_i as i,p.modelMag_z as z,
        p.modelMagErr_u as u_err, p.modelMagErr_g as g_err,
        p.modelMagErr_r as r_err,p.modelMagErr_i as i_err,p.modelMagErr_z as z_err,

        p.MODELMAGERR_U,p.MODELMAGERR_G,p.MODELMAGERR_R,p.MODELMAGERR_I,p.MODELMAGERR_Z,

        p.EXTINCTION_U, p.EXTINCTION_G, p.EXTINCTION_R, p.EXTINCTION_I, p.EXTINCTION_Z,
        p.DERED_U,p.DERED_G,p.DERED_R,p.DERED_I,p.DERED_Z,

        p.PETRORAD_U,p.PETRORAD_G,p.PETRORAD_R,p.PETRORAD_I,p.PETRORAD_Z,
        p.PETRORADERR_U,p.PETRORADERR_G,p.PETRORADERR_R,p.PETRORADERR_I,p.PETRORADERR_Z,

        p.DEVRAD_U,p.DEVRADERR_U,p.DEVRAD_G,p.DEVRADERR_G,p.DEVRAD_R,p.DEVRADERR_R,
        p.DEVRAD_I,p.DEVRADERR_I,p.DEVRAD_Z,p.DEVRADERR_Z,
        p.DEVAB_U,p.DEVAB_G,p.DEVAB_R,p.DEVAB_I,p.DEVAB_Z,

        p.CMODELMAG_U, p.CMODELMAGERR_U, p.CMODELMAG_G,p.CMODELMAGERR_G,
        p.CMODELMAG_R, p.CMODELMAGERR_R, p.CMODELMAG_I,p.CMODELMAGERR_I,
        p.CMODELMAG_Z, p.CMODELMAGERR_Z,

        p.PSFMAG_U, p.PSFMAGERR_U, p.PSFMAG_G, p.PSFMAGERR_G,
        p.PSFMAG_R, p.PSFMAGERR_R, p.PSFMAG_I, p.PSFMAGERR_I,
        p.PSFMAG_Z, p.PSFMAGERR_Z,

        p.FIBERMAG_U, p.FIBERMAGERR_U, p.FIBERMAG_G, p.FIBERMAGERR_G,
        p.FIBERMAG_R, p.FIBERMAGERR_R, p.FIBERMAG_I, p.FIBERMAGERR_I,
        p.FIBERMAG_Z, p.FIBERMAGERR_Z,

        p.FRACDEV_U, p.FRACDEV_G, p.FRACDEV_R, p.FRACDEV_I, p.FRACDEV_Z,
        p.Q_U,p.U_U, p.Q_G,p.U_G, p.Q_R,p.U_R, p.Q_I,p.U_I, p.Q_Z,p.U_Z,

        p.EXPAB_U, p.EXPRAD_U, p.EXPPHI_U, p.EXPAB_G, p.EXPRAD_G, p.EXPPHI_G,
        p.EXPAB_R, p.EXPRAD_R, p.EXPPHI_R, p.EXPAB_I, p.EXPRAD_I, p.EXPPHI_I,
        p.EXPAB_Z, p.EXPRAD_Z, p.EXPPHI_Z,

        p.FIBER2MAG_R, p.FIBER2MAGERR_R,
        p.EXPMAG_R, p.EXPMAGERR_R,

        p.PETROR50_R, p.PETROR90_R, p.PETROMAG_R,
        p.expMag_r + 2.5*log10(2*PI()*p.expRad_r*p.expRad_r + 1e-20) as SB_EXP_R,
        p.petroMag_r + 2.5*log10(2*PI()*p.petroR50_r*p.petroR50_r) as SB_PETRO_R,

        ISNULL(w.j_m_2mass,9999) as J, ISNULL(w.j_msig_2mass,9999) as JERR,
        ISNULL(w.H_m_2mass,9999) as H, ISNULL(w.h_msig_2mass,9999) as HERR,
        ISNULL(w.k_m_2mass,9999) as K, ISNULL(w.k_msig_2mass,9999) as KERR,

        s.survey,
        ISNULL(s.z, -1) as SPEC_Z, ISNULL(s.zErr, -1) as SPEC_Z_ERR, ISNULL(s.zWarning, -1) as SPEC_Z_WARN,
        ISNULL(pz.z,-1) as PHOTOZ, ISNULL(pz.zerr,-1) as PHOTOZ_ERR

        FROM dbo.fGetNearbyObjEq({ra:.10g}, {dec:.10g}, {r_arcmin:.10g}) n, PhotoPrimary p
        INTO mydb.{db_table_name}
        LEFT JOIN SpecObj s ON p.objID = s.bestObjID
        LEFT JOIN PHOTOZ  pz ON p.ObjID = pz.ObjID
        LEFT join WISE_XMATCH as wx on p.objid = wx.sdss_objid
        LEFT join wise_ALLSKY as w on  wx.wise_cntr = w.cntr
        WHERE n.objID = p.objID
    """

    _spec_query_template = """
        SELECT s.specObjID as OBJID, s.bestObjID, s.ra as RA, s.dec as DEC, s.survey,
        ISNULL(s.z, -1) as SPEC_Z, ISNULL(s.zErr, -1) as SPEC_Z_ERR, ISNULL(s.zWarning, -1) as SPEC_Z_WARN
        FROM dbo.fGetNearbySpecObjEq({ra:.10g}, {dec:.10g}, {r_arcmin:.10g}) n, SpecObj s
        INTO mydb.{db_table_name}
        WHERE n.specObjID = s.specObjID
    """

    def __init__(
        self,
        ra,
        dec,
        radius=1.0,
        db_table_name=None,
        context="DR14",
        user=None,
        password=None,
        default_use_sciserver=True,
        specs_only=False,
        sciserver_via_csv=False,
    ):

        self.sciserver_user = user or os.getenv("SCISERVER_USER")
        self.sciserver_pass = password or os.getenv("SCISERVER_PASS")
        self.casjobs_user = user or os.getenv("CASJOBS_WSID")
        self.casjobs_pass = password or os.getenv("CASJOBS_PW")

        self.use_sciserver = False
        if default_use_sciserver and _HAS_SCISERVER_ and self.sciserver_user and self.sciserver_pass:
            self.use_sciserver = True

        if self.use_sciserver:
            self.db_table_name = None
        else:
            if not db_table_name:
                db_table_name = "SAGA" + get_random_string(4)
            self.db_table_name = re.sub("[^A-Za-z]", "", db_table_name)
        self.query = self.construct_query(ra, dec, radius, self.db_table_name, specs_only)
        self.context = context
        self.sciserver_via_csv = sciserver_via_csv

    def download_as_file(self, file_path, overwrite=False, compress=True):
        if os.path.isfile(file_path) and not overwrite:
            return

        if self.use_sciserver:
            self.run_casjobs_with_sciserver(
                self.query,
                file_path,
                compress=compress,
                context=self.context,
                username=self.sciserver_user,
                password=self.sciserver_pass,
                via_csv=self.sciserver_via_csv,
            )
        else:
            self.run_casjobs_with_casjobs(
                self.query,
                self.db_table_name,
                file_path,
                compress=compress,
                context=self.context,
                userid=self.casjobs_user,
                password=self.casjobs_pass,
            )

    @classmethod
    def construct_query(cls, ra, dec, radius=1.0, db_table_name=None, specs_only=False):
        """
        Generates the query to send to the SDSS to get the full SDSS catalog around
        a target.

        Parameters
        ----------
        db_table_name : string
        ra : `Quantity` or float
            The center/host RA (in degrees if float)
        dec : `Quantity` or float
            The center/host Dec (in degrees if float)
        radius : `Quantity` or float
            The radius to search out to (in degrees if float)

        Returns
        -------
        query : str
            The SQL query to send to the SDSS skyserver
        """

        params = {
            "ra": ensure_deg(ra),
            "dec": ensure_deg(dec),
            "r_arcmin": ensure_deg(radius) * 60.0,
            "db_table_name": (db_table_name or "__TO_BE_REMOVED__"),
        }
        query_template = cls._spec_query_template if specs_only else cls._query_template
        q = query_template.format(**params)
        q = re.sub(r"\s+", " ", q).strip()
        q = q.replace("INTO mydb.__TO_BE_REMOVED__", "")
        return q

    @staticmethod
    def run_casjobs_with_casjobs(
        query,
        db_table_name,
        output_path,
        compress=True,
        context="DR14",
        userid=None,
        password=None,
    ):
        """
        Run a single casjobs and download casjobs output using Dan FM's casjobs

        Parameters
        ----------
        query : str, output from construct_query
        db_table_name : str
        output_path : str
        compress : bool, optional
        context : str, optional
        userid : str or int, optional
        password : str, optional

        Notes
        -----
        Follow these instructions to use `run_casjobs_with_casjobs`

        1. Install by Dan FM's casjobs (https://github.com/dfm/casjobs):
            pip install casjobs

        2. Get an account from  http://skyserver.sdss3.org/CasJobs/CreateAccount.aspx

        3. Edit your `.bashrc`:
            export CASJOBS_WSID='2090870927'   # get your WSID from site above
            export CASJOBS_PW='my password'
        """
        if not _HAS_CASJOBS_:
            raise ValueError("Please install casjobs.")

        cjob = casjobs.CasJobs(
            userid,
            password,
            "http://skyserver.sdss.org/casjobs/services/jobs.asmx",
            "POST",
        )

        job_id = cjob.submit(query, context=context, task_name="casjobs_" + db_table_name, estimate=1)
        print(
            time.strftime("[%m/%d %H:%M:%S]"),
            "casjob ({}) submitted...".format(db_table_name),
        )

        code, _ = cjob.monitor(job_id)
        if code == 3 or code == 4:
            raise RuntimeError(
                "{} casjob ({}) {}!".format(
                    time.strftime("[%m/%d %H:%M:%S]"),
                    db_table_name,
                    "cancelled" if code == 3 else "failed",
                )
            )
        assert code == 5

        print(
            time.strftime("[%m/%d %H:%M:%S]"),
            "casjob ({}) finished, downloading data...".format(db_table_name),
        )
        file_open = gzip.open if compress else open
        with file_open(output_path, "wb") as f_out:
            try:
                cjob.request_and_get_output(db_table_name, "FITS", f_out)
            finally:
                cjob.drop_table(db_table_name)

    @staticmethod
    def run_casjobs_with_sciserver(
        query,
        output_path,
        compress=True,
        context="DR14",
        username=None,
        password=None,
        via_csv=False,
    ):
        """
        Run a single casjobs and download casjobs output using SciServer

        Parameters
        ----------
        query : str, output from construct_query
        output_path : str
        compress : bool, optional
        context : str, optional
        username : str, optional
        password : str, optional

        Notes
        -----
        Follow these instructions to use `run_casjobs_with_sciserver`

        1. Install SciServer:
           follow the instruction at https://github.com/sciserver/SciScript-Python

        2. Register an account at https://portal.sciserver.org/login-portal/Account/Register

        3. Edit your `.bashrc`:
            export SCISERVER_USER='username'
            export SCISERVER_PASS='password'
        """
        username = username or os.getenv("SCISERVER_USER")
        password = password or os.getenv("SCISERVER_PASS")
        if not (_HAS_SCISERVER_ and username and password):
            raise ValueError("You are not setup to run casjobs with SciServer")
        SciServer.Authentication.login(username, password)
        return_format = "csv" if via_csv else "fits"
        r = SciServer.CasJobs.executeQuery(query, context=context, format=return_format)
        file_open = gzip.open if compress else open
        with file_open(output_path, "wb") as f_out:
            if via_csv:
                ascii.read(r, format="csv").write(f_out, format="fits")
            else:
                shutil.copyfileobj(r, f_out)


class DatalabQuery(DownloadableBase):

    _query_template = None  # to be implemented by subclass

    def __init__(self, *args, **kwargs):
        self.query = None
        if self._query_template:
            self.set_query(self.construct_query(*args, **kwargs))

    @classmethod
    def construct_query(cls, ra, dec, radius=1.0, **kwargs):
        """
        Generates the query to send to the DES to get the full DES catalog around
        a target.

        Parameters
        ----------
        ra : `Quantity` or float
            The center/host RA (in degrees if float)
        dec : `Quantity` or float
            The center/host Dec (in degrees if float)
        radius : `Quantity` or float
            The radius to search out to (in degrees if float)

        Returns
        -------
        query : str
            The SQL query to send to the Datalab skyserver
        """
        # pylint: disable=possibly-unused-variable
        ra = ensure_deg(ra)
        dec = ensure_deg(dec)
        r_deg = ensure_deg(radius)

        # ``**locals()`` means "use the local variable names to fill the template"
        q = cls._query_template.format(**locals())
        q = re.sub(r"\s+", " ", q).strip()
        q = re.sub(", ", ",", q)
        return q

    def set_query(self, query):
        self.query = str(query)

    def get_colnames_from_query(self, query=None):
        if query is None:
            query = self.query

        cols = []
        for item in query.lower().partition(" from ")[0].split(","):
            _, has_as, name = item.rpartition(" as ")
            if not has_as:
                _, has_dot, name = item.rpartition(".")
                if not has_dot:
                    name = item
            cols.append(name.strip())
        return cols

    def run_query(self, query=None):
        if query is None:
            query = self.query

        if _HAS_DATALAB_:
            raw = dl.queryClient.query(sql=query)
            return Table.read(raw, format="ascii.fast_csv")

        cols = self.get_colnames_from_query(query)

        # taken from https://github.com/noaodatalab/datalab/blob/master/dl/queryClient.py#L1791
        r = requests.get(
            "https://datalab.noao.edu/query/query",
            {"sql": query, "ofmt": "ascii", "async": False},
            headers={
                "Content-Type": "application/octet-stream",
                "X-DL-AuthToken": "anonymous.0.0.anon_access",
            },
            timeout=(120, 3600),
        )
        if not r.ok or not r.text:
            raise requests.RequestException('DES query failed: "{}"'.format(r.text))

        t = Table.read(r.text, format="ascii.fast_tab", names=cols)
        r.close()

        return t

    def download_as_file(self, file_path, overwrite=False, compress=True):
        if os.path.isfile(file_path) and not overwrite:
            return
        t = self.run_query()
        file_open = gzip.open if compress else open
        makedirs_if_needed(file_path)
        with file_open(file_path, "wb") as f:
            t.write(f, format="fits")


class DesQuery(DatalabQuery):
    _query_template = """select
        d.COADD_OBJECT_ID as OBJID,
        d.ALPHAWIN_J2000 as RA,
        d.DELTAWIN_J2000 as DEC,
        d.FLUX_RADIUS_G * 0.263 as radius_g,
        d.FLUX_RADIUS_R * 0.263 as radius_r,
        d.FLUX_RADIUS_I * 0.263 as radius_i,
        d.MAG_AUTO_G_DERED as g_mag,
        d.MAG_AUTO_R_DERED as r_mag,
        d.MAG_AUTO_I_DERED as i_mag,
        d.MAG_AUTO_Z_DERED as z_mag,
        d.MAG_AUTO_Y_DERED as y_mag,
        d.MAGERR_AUTO_G as g_err,
        d.MAGERR_AUTO_R as r_err,
        d.MAGERR_AUTO_I as i_err,
        d.MAGERR_AUTO_Z as z_err,
        d.MAGERR_AUTO_Y as y_err,
        d.A_IMAGE,
        d.B_IMAGE,
        d.THETA_J2000,
        d.flags_r,
        d.imaflags_iso_r,
        (CASE WHEN d.wavg_spread_model_r=-99 THEN d.spread_model_r ELSE d.wavg_spread_model_r END) as spread_model_r,
        (CASE WHEN d.wavg_spreaderr_model_r=-99 THEN d.spreaderr_model_r ELSE d.wavg_spreaderr_model_r END) as spreaderr_model_r
        from des_dr1.main d where
        q3c_radial_query(d.ra, d.dec, {ra:.7g}, {dec:.7g}, {r_deg:.7g})"""


class DelveQuery(DatalabQuery):
    _query_template = """select
        d.quick_object_id as OBJID,
        d.ra as RA,
        d.dec as DEC,
        (d.mag_auto_g - d.extinction_g) as g_mag,
        (d.mag_auto_r - d.extinction_r) as r_mag,
        (d.mag_auto_i - d.extinction_i) as i_mag,
        (d.mag_auto_z - d.extinction_z) as z_mag,
        d.magerr_auto_g as g_err,
        d.magerr_auto_r as r_err,
        d.magerr_auto_i as i_err,
        d.magerr_auto_z as z_err,
        d.a_image_r as radius,
        (d.b_image_r / d.a_image_r) as ba,
        d.theta_image_r as phi,
        d.extended_class_r as morphology_info,
        (CASE WHEN d.extended_class_r=3 THEN 1 ELSE 0 END) as is_galaxy,
        (d.flags_g * 100 + d.flags_r) as REMOVE
        from delve_dr1.objects d where
        q3c_radial_query(d.ra, d.dec, {ra:.7g}, {dec:.7g}, {r_deg:.7g})
        and d.mag_auto_r < 23.5"""


class DecalsPrebuilt(DownloadableBase):

    requires_host_id = True

    def __init__(self, ra, dec, host_id):  # pylint: disable=unused-argument
        try:
            host_id = int(host_id)
        except ValueError:
            self.host_id = host_id
        else:
            self.host_id = "nsa{}".format(host_id)

        self.data_release = "dr6" if ensure_deg(dec) > 32 else "dr7"

    def download_as_file(self, file_path, overwrite=False, compress=True):
        if not compress:
            raise ValueError("Only support compress=True!")
        if os.path.isfile(file_path) and not overwrite:
            return
        r = requests.get(
            "http://www.slac.stanford.edu/~yymao/saga/base-catalogs-non-sdss/{}_decals_{}.fits.gz".format(
                self.host_id, self.data_release
            ),
            headers={"Content-Type": "application/gzip"},
            stream=True,
            timeout=(120, 3600),
        )

        if not r.ok:
            raise requests.RequestException("Decals-prebuilt download failed: '{}'".format(r.text))

        makedirs_if_needed(file_path)
        chunk_size = 16 * 1024 * 1024
        with open(file_path, "wb") as f:
            # here we don't use iter_content because we want to keep gzipped
            while True:
                chunk = r.raw.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        r.close()


class DecalsQuery(DownloadableBase):
    def __init__(
        self,
        ra,
        dec,
        radius=1.0,
        decals_dr="dr9",
        decals_base_dir="/global/project/projectdirs/cosmo/data/legacysurvey",
    ):

        decals_dr = str(decals_dr).lower()
        if not decals_dr.startswith("dr"):
            decals_dr = "dr" + decals_dr

        try:
            dr_number = int(decals_dr[2:4])
        except ValueError:
            try:
                dr_number = int(decals_dr[2])
            except ValueError:
                raise ValueError("Cannot recognize `decals_dr` specification:", decals_dr)

        if dr_number not in (6, 7, 8, 9):
            raise ValueError("{} not supported".format(decals_dr))

        sweep_dir = os.path.join(decals_base_dir, decals_dr, "sweep", "{}.0".format(dr_number))

        if dr_number >= 8:
            self.sweep_dirs = [
                sweep_dir.replace("/sweep/", "/north/sweep/"),
                sweep_dir.replace("/sweep/", "/south/sweep/"),
            ]
        else:
            self.sweep_dirs = [sweep_dir]

        for sweep_dir in self.sweep_dirs:
            if not os.path.isdir(sweep_dir):
                warnings.warn("Cannot access sweep directory {}".format(sweep_dir))
                self.sweep_dirs.remove(sweep_dir)
        if not self.sweep_dirs:
            raise ValueError("No DECaLS sweep directory found! Abort!")

        self.ra = ensure_deg(ra)
        self.dec = ensure_deg(dec)
        self.radius = ensure_deg(radius)

    @staticmethod
    def brickname_to_ra_dec(brickname):
        ra, _, dec = brickname.partition("p" if "p" in brickname else "m")
        ra = float(ra)
        dec = float(dec) * (1 if "p" in brickname else -1)
        return ra, dec

    @classmethod
    def get_ra_dec_range(cls, filename):
        bmin, bmax = filename.partition(".")[0].split("-")[1:]
        ra_min, dec_min = cls.brickname_to_ra_dec(bmin)
        ra_max, dec_max = cls.brickname_to_ra_dec(bmax)
        return ra_min, ra_max, dec_min, dec_max

    def get_decals_catalog(self):
        center_coord = SkyCoord(self.ra, self.dec, unit="deg")
        output = []
        for sweep_dir in self.sweep_dirs:
            for filename in sorted(os.listdir(sweep_dir)):
                if not filename.startswith("sweep-") or not filename.endswith(".fits"):
                    continue
                if not is_within(self.ra, self.dec, *self.get_ra_dec_range(filename), margin=self.radius):
                    continue
                d = FitsTable(os.path.join(sweep_dir, filename)).read()
                mask = SkyCoord(d["RA"], d["DEC"], unit="deg").separation(center_coord).deg <= self.radius
                if mask.any():
                    output.append(d[mask])
                del d, mask

        if not output:
            return Table()

        return vstack(output, "exact")

    def download_as_file(self, file_path, overwrite=False, compress=True):
        if os.path.isfile(file_path) and not overwrite:
            return
        f = FitsTable(file_path)
        f.compress_after_write = bool(compress)
        f.write(self.get_decals_catalog())


class GaiaQuery(DownloadableBase):
    def __init__(self, ra, dec, radius=1.0):
        self.coord = SkyCoord(ra, dec, unit="deg")
        self.radius = radius * u.deg  # pylint: disable=no-member

    def get_gaia_catalog(self):
        if not _HAS_GAIA_:
            raise RuntimeError("Needs astroquery to access Gaia!")
        return Gaia.cone_search_async(self.coord, self.radius).get_data()

    def download_as_file(self, file_path, overwrite=False, **kwargs):
        self.get_gaia_catalog().write(file_path, format="ascii.ecsv", overwrite=overwrite)


def download_catalogs_for_hosts(
    hosts,
    query_class,
    file_path_pattern,
    overwrite=False,
    compress=True,
    file_size_check=1e6,
    host_id_label="HOSTID",
    host_ra_label="RA",
    host_dec_label="DEC",
    **query_class_kwargs,
):
    """
    A convenience function of getting all catalogs for hosts.

    Examples
    --------
    >>> hosts = saga_host_catalog.load()
    >>> file_path_pattern = '/path/to/SAGA/wise/nsa{}.fits.gz'
    >>> failed = download_catalogs_for_hosts(hosts, SdssQuery, file_path_pattern, context='DR14')

    You then can try again for failed ones:
    >>> failed = download_catalogs_for_hosts(hosts[failed], SdssQuery, file_path_pattern, context='DR14')


    Parameters
    ----------
    hosts : astropy.table.Table
    query_class : SdssQuery or WiseQuery
    file_path_pattern : str
    overwrite : bool, optional
    compress : bool, optional
    file_size_check : int, optional
    host_id_label : str, optional
    host_ra_label : str, optional
    host_dec_label : str, optional
    **query_class_kwargs : passed to query_class

    Returns
    -------
    failed : np.array
    """
    failed = np.zeros(len(hosts), np.bool)

    for i, host in enumerate(hosts):
        host_id = host[host_id_label]
        host_ra = host[host_ra_label]
        host_dec = host[host_dec_label]
        path = file_path_pattern.format(host_id)

        print(
            time.strftime("[%m/%d %H:%M:%S]"),
            "Getting catalog for host {} ...".format(host_id),
        )

        if getattr(query_class, "requires_host_id", False):
            query_class_kwargs["host_id"] = host_id

        query_obj = query_class(host_ra, host_dec, **query_class_kwargs)

        try:
            query_obj.download_as_file(path, overwrite=overwrite, compress=compress)
        except Exception as e:
            print(e)
            print(
                time.strftime("[%m/%d %H:%M:%S]"),
                "Fail to get catalog for host {}".format(host_id),
            )
            failed[i] = True
        finally:
            if os.path.isfile(path) and os.path.getsize(path) < file_size_check:
                print(
                    time.strftime("[%m/%d %H:%M:%S]"),
                    "Downloaded catalog corrupted for host {} !!".format(host_id),
                )
                os.unlink(path)
                failed[i] = True

    return failed
