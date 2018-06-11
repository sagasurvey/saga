import os
import time
import re
import gzip
import random
import string
import shutil
import requests
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from .core import FitsTable
from ..utils import makedirs_if_needed

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


__all__ = ['SdssQuery', 'WiseQuery', 'DesQuery', 'DecalsQuery', 'download_catalogs_for_hosts']


def get_random_string(length=6):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def ensure_deg(value):
    if isinstance(value, u.Quantity):
        return value.to(u.deg).value # pylint: disable=E1101
    return float(value)


class WiseQuery(FitsTable):
    """
    Examples
    --------
    WiseQuery(ra, dec).download_as_file('/path/to/file', overwrite=False)

    """
    def __init__(self, ra, dec, radius=1.0, **kwargs):
        path = 'http://unwise.me/phot_near/?ra={:f}&dec={:f}&radius={:f}&datatype=flat&version=sdss-dr10d'.format(ensure_deg(ra), ensure_deg(dec), ensure_deg(radius))
        super(WiseQuery, self).__init__(path, **kwargs)

    def write(self, table):
        raise NotImplementedError


class SdssQuery(object):
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

    def __init__(self, ra, dec, radius=1.0, db_table_name=None, context='DR14',
                 user=None, password=None, default_use_sciserver=True):

        self.sciserver_user = user or os.getenv('SCISERVER_USER')
        self.sciserver_pass = password or os.getenv('SCISERVER_PASS')
        self.casjobs_user = user or os.getenv('CASJOBS_WSID')
        self.casjobs_pass = password or os.getenv('CASJOBS_PW')

        self.use_sciserver = False
        if default_use_sciserver and _HAS_SCISERVER_ and self.sciserver_user and self.sciserver_pass:
            self.use_sciserver = True

        if self.use_sciserver:
            self.db_table_name = None
        else:
            if not db_table_name:
                db_table_name = 'SAGA' + get_random_string(4)
            self.db_table_name = re.sub('[^A-Za-z]', '', db_table_name)
        self.query = self.construct_query(ra, dec, radius, self.db_table_name)
        self.context = context


    def download_as_file(self, file_path, overwrite=False, compress=True):
        if os.path.isfile(file_path) and not overwrite:
            return

        if self.use_sciserver:
            self.run_casjobs_with_sciserver(self.query, file_path, compress=compress, context=self.context, username=self.sciserver_user, password=self.sciserver_pass)
        else:
            self.run_casjobs_with_casjobs(self.query, self.db_table_name, file_path, compress=compress, context=self.context, userid=self.casjobs_user, password=self.casjobs_pass)


    @classmethod
    def construct_query(cls, ra, dec, radius=1.0, db_table_name=None):
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

        select_into_mydb = True
        if db_table_name is None:
            db_table_name = 'TO_BE_REMOVED'
            select_into_mydb = False

        ra = ensure_deg(ra)
        dec = ensure_deg(dec)
        r_arcmin = ensure_deg(radius) * 60.0

        # ``**locals()`` means "use the local variable names to fill the template"
        q = cls._query_template.format(**locals())
        q = re.sub(r'[^\S\n]+', ' ', q).strip()
        if not select_into_mydb:
            q = q.replace('INTO mydb.{} '.format(db_table_name), '')
        return q


    @staticmethod
    def run_casjobs_with_casjobs(query, db_table_name, output_path, compress=True, context='DR14',
                                 userid=None, password=None):
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
            raise ValueError('Please install casjobs.')

        cjob = casjobs.CasJobs(userid, password, 'http://skyserver.sdss.org/casjobs/services/jobs.asmx', 'POST')

        job_id = cjob.submit(query, context=context, task_name='casjobs_'+db_table_name, estimate=1)
        print(time.strftime('[%m/%d %H:%M:%S]'), 'casjob ({}) submitted...'.format(db_table_name))

        code, status = cjob.monitor(job_id)
        if code == 3 or code == 4:
            raise RuntimeError('{} casjob ({}) {}!'.format(time.strftime('[%m/%d %H:%M:%S]'), db_table_name, 'cancelled' if code == 3 else 'failed'))
        assert code == 5

        print(time.strftime('[%m/%d %H:%M:%S]'), 'casjob ({}) finished, downloading data...'.format(db_table_name))
        file_open = gzip.open if compress else open
        with file_open(output_path, 'wb') as f_out:
            try:
                cjob.request_and_get_output(db_table_name, 'FITS', f_out)
            finally:
                cjob.drop_table(db_table_name)


    @staticmethod
    def run_casjobs_with_sciserver(query, output_path, compress=True, context='DR14',
                                   username=None, password=None):
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
        username = username or os.getenv('SCISERVER_USER')
        password = password or os.getenv('SCISERVER_PASS')
        if not (_HAS_SCISERVER_ and username and password):
            raise ValueError('You are not setup to run casjobs with SciServer')
        SciServer.Authentication.login(username, password)
        r = SciServer.CasJobs.executeQuery(query, context=context, format="fits")
        file_open = gzip.open if compress else open
        with file_open(output_path, 'wb') as f_out:
            shutil.copyfileobj(r, f_out)


class DesQuery(object):

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
        d.flags_r,
        d.imaflags_iso_r,
        (CASE WHEN (d.wavg_spread_model_i + 3*d.wavg_spreaderr_model_i) > 0.005 THEN 1 ELSE 0 END) +
        (CASE WHEN (d.wavg_spread_model_i + d.wavg_spreaderr_model_i) > 0.003 THEN 1 ELSE 0 END) +
        (CASE WHEN (d.wavg_spread_model_i - d.wavg_spreaderr_model_i) > 0.003 THEN 1 ELSE 0 END) as wavg_extended_coadd_i
        from des_dr1.main d where
        q3c_radial_query(d.ra, d.dec, {ra:.7g}, {dec:.7g}, {r_deg:.7g})"""

    def __init__(self, ra, dec, radius=1.0):
        self.query = self.construct_query(ra, dec, radius)

    @classmethod
    def construct_query(cls, ra, dec, radius=1.0):
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
            The SQL query to send to the SDSS skyserver
        """
        ra = ensure_deg(ra)
        dec = ensure_deg(dec)
        r_deg = ensure_deg(radius)

        # ``**locals()`` means "use the local variable names to fill the template"
        q = cls._query_template.format(**locals())
        q = re.sub(r'\s+', ' ', q).strip()
        q = re.sub(', ', ',', q)
        return q

    def download_as_file(self, file_path, overwrite=False, compress=True):
        if os.path.isfile(file_path) and not overwrite:
            return

        r = requests.get('https://dlsvcs.datalab.noao.edu/query/query',
                         {'sql': self.query, 'ofmt': 'fits', 'async': False},
                         headers={'Content-Type': 'application/octet-stream', 'X-DL-AuthToken': 'anonymous.0.0.anon_access'},
                         stream=True)

        r.raw.decode_content = True
        file_open = gzip.open if compress else open
        with file_open(file_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


class DecalsPrebuilt(object):

    requires_host_id = True

    def __init__(self, ra, dec, host_id):
        try:
            host_id = int(host_id)
        except ValueError:
            self.host_id = host_id
        else:
            self.host_id = 'nsa{}'.format(host_id)

        self.data_release = 'dr6' if ensure_deg(dec) > 32 else 'dr5'

    def download_as_file(self, file_path, overwrite=False, compress=True):
        if not compress:
            raise ValueError('Only support compress=True!')
        if os.path.isfile(file_path) and not overwrite:
            return
        r = requests.get('http://www.slac.stanford.edu/~yymao/saga/base-catalogs-non-sdss/{}_decals_{}.fits.gz'.format(self.host_id, self.data_release),
                         headers={'Content-Type': 'application/gzip'},
                         stream=True)
        makedirs_if_needed(file_path)
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


class DecalsQuery(object):

    columns_needed = "RELEASE BRICKID OBJID TYPE RA DEC FLUX_G FLUX_R FLUX_Z FLUX_IVAR_G FLUX_IVAR_R FLUX_IVAR_Z MW_TRANSMISSION_G MW_TRANSMISSION_R MW_TRANSMISSION_Z NOBS_G NOBS_R NOBS_Z RCHISQ_G RCHISQ_R RCHISQ_Z FRACMASKED_G FRACMASKED_R FRACMASKED_Z ALLMASK_G ALLMASK_R ALLMASK_Z FRACDEV FRACDEV_IVAR SHAPEDEV_R SHAPEDEV_R_IVAR SHAPEEXP_R SHAPEEXP_R_IVAR".split()

    def __init__(self, ra, dec, radius=1.0, decals_dr='dr5',
                 decals_base_dir='/global/project/projectdirs/cosmo/data/legacysurvey'):

        self.sweep_dir = os.path.join(decals_base_dir, decals_dr, 'sweep', decals_dr[-1]+'.0')
        if not os.path.isdir(self.sweep_dir):
            raise ValueError('DECaLS sweep directory {} does not exist!'.format(self.sweep_dir))

        self.ra = ensure_deg(ra)
        self.dec = ensure_deg(dec)
        self.radius = ensure_deg(radius)

    @staticmethod
    def brickname_to_ra_dec(brickname):
        ra, _, dec = brickname.partition('p' if 'p' in brickname else 'm')
        ra = float(ra)
        dec = float(dec) * (1 if 'p' in brickname else -1)
        return ra, dec

    @classmethod
    def get_ra_dec_range(cls, filename):
        bmin, bmax = filename.partition('.')[0].split('-')[1:]
        ra_min, dec_min = cls.brickname_to_ra_dec(bmin)
        ra_max, dec_max = cls.brickname_to_ra_dec(bmax)
        return ra_min, ra_max, dec_min, dec_max

    @staticmethod
    def is_within(ra, dec, ra_min, ra_max, dec_min, dec_max, margin_ra=0, margin_dec=0):
        return ((ra_min - margin_ra <= ra) & (ra_max + margin_ra >= ra) & (dec_min - margin_dec <= dec) & (dec_max + margin_dec >= dec))

    @staticmethod
    def annotate_catalog(d):
        for band in 'grz':
            BAND = band.upper()
            d[band+'_mag'] = 22.5 - 2.5 * np.log10(d['FLUX_'+BAND] / d['MW_TRANSMISSION_'+BAND])
            d[band+'_err'] = 2.5 / np.log(10) / (d['FLUX_'+BAND] / d['MW_TRANSMISSION_'+BAND]) / np.sqrt(d['FLUX_IVAR_'+BAND])
        return d

    def get_decals_catalog(self):
        output = []
        for filename in sorted(os.listdir(self.sweep_dir)):
            if not filename.startswith('sweep-'):
                continue
            if not self.is_within(self.ra, self.dec, *self.get_ra_dec_range(filename),
                                  margin_ra=self.radius*1.01/max(np.cos(np.deg2rad(self.dec)), 1.0e-8),
                                  margin_dec=self.radius*1.01):
                continue

            d = FitsTable(os.path.join(self.sweep_dir, filename)).read()[self.columns_needed]
            sep = SkyCoord(d['RA'], d['DEC'], unit='deg').separation(SkyCoord(self.ra, self.dec, unit='deg')).deg
            d = d[sep <= self.radius]
            if len(d):
                output.append(d)

        del d
        if not output:
            return Table()

        output = vstack(output, 'exact', 'error')
        return self.annotate_catalog(output)

    def download_as_file(self, file_path, overwrite=False, compress=True):
        if os.path.isfile(file_path) and not overwrite:
            return
        f = FitsTable(file_path)
        f.compress_after_write = bool(compress)
        f.write(self.get_decals_catalog())


def download_catalogs_for_hosts(hosts, query_class, file_path_pattern,
                                overwrite=False, compress=True, file_size_check=1e6,
                                host_id_label='NSAID', host_ra_label='RA', host_dec_label='Dec',
                                **query_class_kwargs):
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

        print(time.strftime('[%m/%d %H:%M:%S]'), 'Getting catalog for host {} ...'.format(host_id))

        if getattr(query_class, 'requires_host_id', False):
            query_class_kwargs['host_id'] = host_id

        query_obj = query_class(host_ra, host_dec, **query_class_kwargs)

        try:
            query_obj.download_as_file(path, overwrite=overwrite, compress=compress)
        except (IOError, OSError, RuntimeError, requests.RequestException) as e:
            print(e)
            print(time.strftime('[%m/%d %H:%M:%S]'), 'Fail to get catalog for host {}'.format(host_id))
            failed[i] = True
        else:
            if os.path.getsize(path) < file_size_check:
                print(time.strftime('[%m/%d %H:%M:%S]'), 'Downloaded catalog corrupted for host {} !!'.format(host_id))
                os.unlink(path)
                failed[i] = True

    return failed
