"""
TargetSelection class
"""
from itertools import chain
import numpy as np
from easyquery import Query
from astropy.table import vstack, Table
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from ..hosts import HostCatalog
from ..objects import ObjectCatalog, build
from .assign_targeting_score import assign_targeting_score_v1, assign_targeting_score_v2, COLUMNS_USED

__all__ = ['TargetSelection', 'prepare_mmt_catalog', 'prepare_aat_catalog']


class TargetSelection(object):
    """
    Parameters
    ----------
    database: SAGA.Database object

    Returns
    -------
    target_selection : SAGA.TargetSelection object

    Examples
    --------
    >>> import SAGA
    >>> from SAGA import ObjectCuts as C
    >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox')
    >>> saga_targets = SAGA.TargetSelection(saga_database, gmm_parameters='gmm_parameters_no_outlier')
    >>> hosts = [161174, 52773, 163956, 69028, 144953, 165082, 165707, 145729, 165980, 147606]
    >>> saga_targets.load_object_catalogs(hosts, (C.gri_cut & C.fibermag_r_cut & C.is_galaxy & C.is_clean))
    >>> score_bins = [150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    >>> d = np.array([np.searchsorted(base['TARGETING_SCORE'], score_bins) for base in saga_targets.compile_target_list('iter')])
    """
    def __init__(self, database, cuts=None, additional_columns=None,
                 assign_targeting_score_func=None, gmm_parameters=None,
                 manual_selected_objids=None, version=2):
        self._version = version
        self._database = database
        self._host_catalog = HostCatalog(self._database)
        self._object_catalog = ObjectCatalog(self._database)

        self.target_catalogs = dict()

        if assign_targeting_score_func is None:
            self._assign_targeting_score = assign_targeting_score_v1 if self._version == 1 else assign_targeting_score_v2
        else:
            self._assign_targeting_score = assign_targeting_score_func
            if not callable(self._assign_targeting_score):
                raise TypeError('*assign_targeting_score_func* must be callable')

        if isinstance(gmm_parameters, dict):
            self._gmm_parameters = {k: self._load_gmm_parameters(v) for k, v in gmm_parameters.items()}
        else:
            self._gmm_parameters = self._load_gmm_parameters(gmm_parameters)

        try:
            self._manual_selected_objids = build.get_unique_objids(self._database[manual_selected_objids or 'manual_targets'].read()['OBJID'])
        except (TypeError, KeyError):
            self._manual_selected_objids = manual_selected_objids

        self._cuts = cuts
        if self._version == 1:
            self.columns = list(set(chain(('OBJID', 'RA', 'DEC', 'HOST_NSAID'),
                                          ('PHOTPTYPE', 'PSFMAG_U', 'PSFMAG_G', 'PSFMAG_R'),
                                          COLUMNS_USED,
                                          additional_columns or [])))
        else:
            self.columns = None
            if additional_columns is not None:
                raise ValueError('`additional_columns` is not supported for version > 1')


    def _load_gmm_parameters(self, gmm_parameters):
        try:
            return self._database[gmm_parameters].read()
        except (TypeError, KeyError):
            return gmm_parameters


    def build_target_catalogs(self, hosts=None, return_as=None, columns=None,
                              reload_base=False, recalculate_score=False):
        """
        build target catalogs

        Parameters
        ----------
        hosts : int, str, list, None, optional
            host names/IDs or a list of host names/IDs or short-hand names like
            "paper1" or "paper1_complete"

        return_as : str, optional
            If set to None (default), no return
            If set to 'dict', return as a dict
            If set to 'list', return a list that contains all tables
            If set to 'stacked', return a stacked table
            If set to 'iter', return an iterator for looping over hosts

        columns : list, optional
            If set, only return a subset of columns
        """
        return_as = (return_as if return_as else 'none').lower()
        if return_as[0] not in 'ndsli':
            raise ValueError('`return_as` should be None, "dict", "list", "stacked", or "iter"')

        host_ids = self._host_catalog.resolve_id(hosts, 'string')

        for host_id in host_ids:
            if reload_base or host_id not in self.target_catalogs:
                self.target_catalogs[host_id] = self._object_catalog.load(host_id, \
                        cuts=self._cuts, columns=self.columns, return_as='list', version=self._version).pop()
                if 'coord' in self.target_catalogs[host_id].colnames:
                    del self.target_catalogs[host_id]['coord']

            if recalculate_score or 'TARGETING_SCORE' not in self.target_catalogs[host_id].colnames:
                self._assign_targeting_score(self.target_catalogs[host_id], self._manual_selected_objids, self._gmm_parameters)

        if return_as[0] != 'n':
            output_iter = (self.target_catalogs[host_id][columns] if columns else self.target_catalogs[host_id] for host_id in host_ids)
            if return_as[0] == 'd':
                return dict(zip(host_ids, output_iter))
            elif return_as[0] == 'i':
                return output_iter
            elif return_as[0] == 'l':
                return list(output_iter)
            elif return_as[0] == 's':
                return vstack(list(output_iter), 'outer', 'error')


    def clear_target_catalogs(self):
        """
        clear target catalog cache
        """
        self.target_catalogs = dict()


def prepare_mmt_catalog(target_catalog, write_to=None, flux_star_removal_threshold=20.0, verbose=True):
    """
    Prepare MMT target catalog.

    Parameters
    ----------
    target_catalog : astropy.table.Table
        Need to have `TARGETING_SCORE` column.
        You can use `TargetSelection.build_target_catalogs` to generate `target_catalog`
    write_to : str, optional
        If set, it will write the catalog in MMT format to `write_to`.
    flux_star_removal_threshold : float, optional
        In arcseconds
    verbose : bool, optional

    Returns
    -------
    mmt_target_catalog : astropy.table.Table

    Examples
    --------
    >>> import SAGA
    >>> from SAGA.targets import prepare_mmt_catalog
    >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox')
    >>> saga_targets = SAGA.TargetSelection(saga_database, gmm_parameters='gmm_parameters_no_outlier')
    >>> mmt18_hosts = [161174, 52773, 163956, 69028, 144953, 165082, 165707, 145729, 165980, 147606]
    >>> for host_id, target_catalog in saga_targets.build_target_catalogs(mmt18_hosts, return_as='dict').items():
    >>>     print('Working host NSA', host_id)
    >>>     SAGA.targets.prepare_mmt_catalog(target_catalog, '/home/yymao/Downloads/mmt_nsa{}.cat'.format(host_id))
    >>>     print()

    Notes
    -----
    See https://www.cfa.harvard.edu/mmti/hectospec/hecto_software_manual.htm#4.1.1 for required format

    """

    if 'TARGETING_SCORE' not in target_catalog.colnames:
        return KeyError('`target_catalog` does not have column "TARGETING_SCORE".'
                        'Have you run `compile_target_list` or `assign_targeting_score`?')

    is_target = Query('TARGETING_SCORE >= 0', 'TARGETING_SCORE < 900')

    is_star = Query('PHOTPTYPE == 6')
    is_guide_star = is_star & Query('PSFMAG_R >= 14', 'PSFMAG_R < 15')
    is_flux_star = is_star & Query('PSFMAG_R >= 17', 'PSFMAG_R < 18')
    is_flux_star &= Query('PSFMAG_U - PSFMAG_G >= 0.6', 'PSFMAG_U - PSFMAG_G < 1.2')
    is_flux_star &= Query('PSFMAG_G - PSFMAG_R >= 0', 'PSFMAG_G - PSFMAG_R < 0.6')
    is_flux_star &= Query('(PSFMAG_G - PSFMAG_R) > 0.75 * (PSFMAG_U - PSFMAG_G) - 0.45')

    target_catalog = (is_target | is_guide_star | is_flux_star).filter(target_catalog)

    target_catalog['rank'] = target_catalog['TARGETING_SCORE'] // 100
    target_catalog['rank'][Query('rank < 2').mask(target_catalog)] = 2 # regular targets start at rank 2
    target_catalog['rank'][is_flux_star.mask(target_catalog)] = 1
    target_catalog['rank'][is_guide_star.mask(target_catalog)] = 99 # set to 99 for sorting

    flux_star_indices = np.flatnonzero(is_flux_star.mask(target_catalog))
    flux_star_sc = SkyCoord(*target_catalog[['RA', 'DEC']][flux_star_indices].itercols(), unit='deg')
    target_sc = SkyCoord(*is_target.filter(target_catalog)[['RA', 'DEC']].itercols(), unit='deg')
    sep = flux_star_sc.match_to_catalog_sky(target_sc)[1]
    target_catalog['rank'][flux_star_indices[sep.arcsec < flux_star_removal_threshold]] = 0
    target_catalog = Query('rank > 0').filter(target_catalog)

    if verbose:
        print('# of guide stars     =', is_guide_star.count(target_catalog))
        print('# of flux stars      =', is_flux_star.count(target_catalog))
        print('# of rank>1 targets  =', is_target.count(target_catalog))
        for rank in range(1, 9):
            print('# of rank={} targets ='.format(rank),
                Query('rank == {}'.format(rank)).count(target_catalog))

    target_catalog['type'] = 'TARGET'
    target_catalog['type'][is_guide_star.mask(target_catalog)] = 'guide'

    target_catalog.rename_column('RA', 'ra')
    target_catalog.rename_column('DEC', 'dec')
    target_catalog.rename_column('OBJID', 'object')
    target_catalog.rename_column('r_mag', 'mag')

    target_catalog.sort(['rank', 'TARGETING_SCORE', 'mag'])
    target_catalog = target_catalog[['ra', 'dec', 'object', 'rank', 'type', 'mag']]

    if write_to:
        if verbose:
            print('Writing to {}'.format(write_to))

        if not write_to.endswith('.cat'):
            print('Warning: filename should end with \'.cat\'')

        with open(write_to, 'w') as fh:
            fh.write('\t'.join(target_catalog.colnames) + '\n')
            # the MMT format is odd and *requires* "---"'s in the second header line
            fh.write('\t'.join(('-'*len(s) for s in target_catalog.colnames)) + '\n')
            target_catalog.write(fh,
                                 delimiter='\t',
                                 format='ascii.fast_no_header',
                                 formats={
                                     'ra': lambda x: Angle(x, 'deg').wrap_at(360*u.deg).to_string('hr', sep=':', precision=3), # pylint: disable=E1101
                                     'dec': lambda x: Angle(x, 'deg').to_string('deg', sep=':', precision=3),
                                     'mag': '%.2f',
                                     'rank': lambda x: '' if x == 99 else '{:d}'.format(x),
                                 })

    return target_catalog


def prepare_aat_catalog(target_catalog, write_to=None, flux_star_removal_threshold=20.0, verbose=True):
    """
    Prepare AAT target catalog.

    Format needed:

    # TargetName(unique for header) RA(h m s) Dec(d m s) TargetType(Program,Fiducial,Sky) Priority(9 is highest) Magnitude 0 Notes
    1237648721248518305 14 42 17.79 -0 12 05.95 P 2 22.03 0 magcol=fiber2mag_r, model_r=20.69
    1237648721786045341 14 48 37.16 +0 21 33.81 P 1 21.56 0 magcol=fiber2mag_r, model_r=20.55
    """

    if 'TARGETING_SCORE' not in target_catalog.colnames:
        return KeyError('`target_catalog` does not have column "TARGETING_SCORE".'
                        'Have you run `compile_target_list` or `assign_targeting_score`?')

    ra_min = target_catalog['RA'].min()
    ra_max = target_catalog['RA'].max()
    sin_dec_min = np.sin(np.deg2rad(target_catalog['DEC'].min()))
    sin_dec_max = np.sin(np.deg2rad(target_catalog['DEC'].max()))

    n_needed = 100
    ra_sky = []
    dec_sky = []
    base_sc = SkyCoord(target_catalog['RA'], target_catalog['DEC'], unit='deg')
    seed = 12
    while n_needed > 0:
        ra_rand = np.random.RandomState(seed).uniform(ra_min, ra_max, size=n_needed)
        dec_rand = np.rad2deg(np.arcsin(np.random.RandomState(seed).uniform(sin_dec_min, sin_dec_max, size=n_needed)))
        sky_sc = SkyCoord(ra_rand, dec_rand, unit='deg')
        sep = sky_sc.match_to_catalog_sky(base_sc)[1]
        ok_mask = (sep.arcsec > 10.0)
        n_needed -= np.count_nonzero(ok_mask)
        ra_sky.append(ra_rand[ok_mask])
        dec_sky.append(dec_rand[ok_mask])
        seed += np.random.RandomState(seed).randint(100, 200)
        del ra_rand, dec_rand, sky_sc, sep, ok_mask
    del base_sc
    ra_sky = np.concatenate(ra_sky)
    dec_sky = np.concatenate(dec_sky)

    is_target = Query('TARGETING_SCORE >= 0', 'TARGETING_SCORE < 900')
    is_des = Query((lambda s: s == 'des', 'survey'))
    is_star = Query('morphology_info == 0', is_des) | Query(~is_des, ~Query('is_galaxy'))
    is_flux_star = Query(is_star, 'r_mag > 16', 'r_mag < 17')

    target_catalog = (is_target | is_flux_star).filter(target_catalog)
    target_catalog['Priority'] = target_catalog['TARGETING_SCORE'] // 100
    target_catalog['Priority'][is_flux_star.mask(target_catalog)] = 9

    flux_star_indices = np.flatnonzero(is_flux_star.mask(target_catalog))
    flux_star_sc = SkyCoord(*target_catalog[['RA', 'DEC']][flux_star_indices].itercols(), unit='deg')
    target_sc = SkyCoord(*is_target.filter(target_catalog)[['RA', 'DEC']].itercols(), unit='deg')
    sep = flux_star_sc.match_to_catalog_sky(target_sc)[1]
    target_catalog['Priority'][flux_star_indices[sep.arcsec < flux_star_removal_threshold]] = -1
    target_catalog = Query('Priority >= 0').filter(target_catalog)
    del flux_star_indices, flux_star_sc, target_sc, sep

    target_catalog['TargetType'] = 'P'
    target_catalog['0'] = 0
    target_catalog['Notes'] = 'targets'

    target_catalog.rename_column('DEC', 'Dec')
    target_catalog.rename_column('OBJID', 'TargetName')
    target_catalog.rename_column('r_mag', 'Magnitude')

    target_catalog.sort(['Priority', 'TARGETING_SCORE', 'Magnitude'])
    target_catalog = target_catalog[['TargetName', 'RA', 'Dec', 'TargetType', 'Priority', 'Magnitude', '0', 'Notes']]

    sky_catalog = Table({
        'TargetName': np.arange(len(ra_sky)),
        'RA': ra_sky,
        'Dec': dec_sky,
        'TargetType': np.repeat('S', len(ra_sky)),
        'Priority': np.repeat(9, len(ra_sky)),
        'Magnitude': np.repeat(99.0, len(ra_sky)),
        '0': np.repeat(0, len(ra_sky)),
        'Notes': np.repeat('sky', len(ra_sky)),
    })

    target_catalog = vstack([target_catalog, sky_catalog])

    if verbose:
        for rank in range(1, 9):
            print('# of Priority={} targets ='.format(rank),
                Query('Priority == {}'.format(rank)).count(target_catalog))

    if write_to:
        if verbose:
            print('Writing to {}'.format(write_to))

        target_catalog.write(write_to,
                             delimiter=' ',
                             quotechar='"',
                             format='ascii.fast_commented_header',
                             overwrite=True,
                             formats={
                                'RA': lambda x: Angle(x, 'deg').wrap_at(360*u.deg).to_string('hr', sep=' ', precision=2), # pylint: disable=E1101
                                'Dec': lambda x: Angle(x, 'deg').to_string('deg', sep=' ', precision=2),
                                'Magnitude': '%.2f',
                             })

        with open(write_to) as fh:
            content = fh.read()

        with open(write_to, 'w') as fh:
            fh.write(content.replace('"', ''))

    return target_catalog
