import numpy as np
from astropy.coordinates import SkyCoord
from astropy.constants import c

def clean_repeats(spectra):

    """
    Clean all spectra to remove repeats.
    `spectra` is modified in-place.

    Parameters
    ----------
    spectra : astropy.table.Table

    Returns
    -------
    spectra : astropy.table.Table
    """

    # make copies of the ra, dec, z, and indices of the whole spectra
    # we need copies as these will be later sliced in place
    spectra_sc = SkyCoord(spectra['RA'], spectra['DEC'], unit="deg")
    spectra_z = np.array(spectra['SPEC_Z'])
    spectra_idx = np.arange(len(spectra))

    for i, spec in enumerate(spectra):
        if i not in spectra_idx:
            continue

        spec_sc = SkyCoord(spec['RA'], spec['DEC'], unit='deg')

        # search nearby spectra in 3D
        nearby_mask = (np.abs(spectra['SPEC_Z'] - spec['SPEC_Z']) < 50.0/c.to('km/s').value)
        nearby_mask &= (spectra_sc.separation(spec_sc).arcsec < 30.0)

        specs_nearby = spectra[spectra_idx[nearby_mask]]

        spectra_sc = spectra_sc[~nearby_mask]
        spectra_z = spectra_z[~nearby_mask]
        spectra_idx = spectra_idx[~nearby_mask]

        done_spectra_indices = []

        spec_repeat = set()
        for r in specs_nearby['SPEC_REPEAT']:
            spec_repeat.update(r.split('+'))
        spec_repeat = '+'.join(spec_repeat)

        #XXX: not finished!!!

        # should prefer NSA
        best_spec = specs_nearby[specs_nearby['ZQUALITY'].data.argmax()]
        best_spec_sc = SkyCoord(best_spec['RA'], best_spec['DEC'], unit='deg')
        closest_object_index = SkyCoord(objects_nearby['RA'], objects_nearby['DEC'], unit='deg').separation(best_spec_sc).arcsec.argmin()

        original_base_index = objects_nearby_indices[closest_object_index]
        base['SPEC_REPEAT'][original_base_index] = spec_repeat
        for col in cols_to_copy:
            base[col.upper()][original_base_index] = best_spec[col]

        other_objects = objects_nearby_indices[objects_nearby_indices != original_base_index]
        base['REMOVE'][other_objects] = 0


    return base
