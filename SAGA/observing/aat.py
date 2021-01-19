import os
from io import StringIO

import numpy as np
import requests
from astropy import table
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord

# pylint: disable=no-member

__all__ = [
    "get_gaia_guidestars",
    "get_sdss_guidestars",
    "write_fld_file",
    "subsample_catalog",
    "make_decals_viewer_cutouts",
    "make_des_cutouts_file",
    "show_des_cutouts",
]


def get_gaia_guidestars(
    hostname=None,
    host_catalog=None,
    object_catalog=None,
    magrng=(12.5, 13.5),
    matchmagrng=(16, 17),
    d_matchmag=1,
    matchtol0=1 * u.arcsec,
    verbose=True,
    neighbor_cut=30 * u.arcsec,
    nmagdown=4.5,
    gaia_catalog=None,
):
    """
    `magrng` is the range of magnitudes to actually select on.  It's ~r-band, based on the
    Evans et al. 2018 r-to-G conversion
    """
    if gaia_catalog is None:
        gaia_cat = table.Table.read("external_catalogs/astrometric/{}_gaia.ecsv".format(hostname))
    else:
        gaia_cat = gaia_catalog

    if isinstance(object_catalog, table.Table):
        obj_cat = object_catalog
    else:
        obj_cat = object_catalog.load(hostname)[0]

    omag = obj_cat["r_mag"]
    gaia_sc = SkyCoord(gaia_cat["ra"], gaia_cat["dec"])
    gmag = gaia_cat["phot_g_mean_mag"]

    omsk = (matchmagrng[0] < omag) & (omag < matchmagrng[1])
    gmsk = ((matchmagrng[0] - d_matchmag) < gmag) & (gmag < (matchmagrng[1] + d_matchmag))
    oscmsk = obj_cat["coord"][omsk]
    gscmsk = gaia_sc[gmsk]

    idx, d2d, _ = oscmsk.match_to_catalog_sky(gscmsk)
    sepmsk = d2d < matchtol0

    dra = (oscmsk[sepmsk].ra - gscmsk[idx][sepmsk].ra).to(u.arcsec)
    ddec = (oscmsk[sepmsk].dec - gscmsk[idx][sepmsk].dec).to(u.arcsec)
    offset = np.mean(dra), np.mean(ddec)

    if verbose:
        print(
            "Object catalog to Gaia offset:",
            offset,
            "from",
            np.sum(sepmsk),
            "objects ({:.1%})".format(np.sum(sepmsk) / len(sepmsk)),
        )

    # this polynomial is from Evans et al. 2018 for the G to r conversion
    Gmr_coeffs = (-0.1856, 0.1579, 0.02738, -0.0550)
    Gmr = np.polyval(Gmr_coeffs[::-1], gaia_cat["bp_rp"])
    gaia_cat["g_as_r_mag"] = gmag - Gmr

    gmsk = (magrng[0] < gaia_cat["g_as_r_mag"]) & (gaia_cat["g_as_r_mag"] < magrng[1])
    gstars = gaia_cat[gmsk]
    print("Found", len(gstars), "Gaia guide stars")

    if neighbor_cut is not None:
        possible_neighbor_stars = gaia_cat[gaia_cat["g_as_r_mag"] < magrng[1] + nmagdown]
        nsc = SkyCoord(possible_neighbor_stars["ra"], possible_neighbor_stars["dec"])
        gsc = SkyCoord(gstars["ra"], gstars["dec"])
        idx, d2d, _ = gsc.match_to_catalog_sky(nsc, 2)
        neighbor_present = d2d < neighbor_cut

        print(np.sum(neighbor_present), "Have a brightish neighbor.  Removing them.")
        gstars = gstars[~neighbor_present]

    tab = table.Table(
        {
            "TargetName": gstars["source_id"],
            "RA": (u.Quantity(gstars["ra"]) + offset[0]),
            "Dec": (u.Quantity(gstars["dec"]) + offset[1]),
            "TargetType": np.repeat("F", len(gstars)),
            "Priority": np.repeat(9, len(gstars)),
            "Magnitude": gstars["g_as_r_mag"],
            "0": np.repeat(0, len(gstars)),
            "Notes": np.repeat("guide_gaia", len(gstars)),
        }
    )
    tab.meta["dra"] = dra
    tab.meta["ddec"] = ddec
    tab.meta["offset"] = offset
    tab.meta["gstarcat"] = gstars
    return tab


def get_sdss_guidestars(hostname, host_catalog, object_catalog, verbose=True):

    obj_cat = object_catalog.load(hostname)[0]
    r = obj_cat["r_mag"]
    msk = (12.5 < r) & (r < 14) & ~obj_cat["is_galaxy"] & (obj_cat["RHOST_ARCM"] > 15)
    starcat = obj_cat[msk]

    if verbose:
        print("Found", len(starcat), "SDSS guide stars")

    return table.Table(
        {
            "TargetName": starcat["OBJID"],
            "RA": starcat["RA"],
            "Dec": starcat["DEC"],
            "TargetType": np.repeat("F", len(starcat)),
            "Priority": np.repeat(9, len(starcat)),
            "Magnitude": starcat["r_mag"],
            "0": np.repeat(0, len(starcat)),
            "Notes": np.repeat("guide", len(starcat)),
        }
    )


def write_fld_file(
    target_catalog,
    host,
    obstime,
    fn,
    suffix=" master catalog",
    host_id_label="HOSTID",
    host_coord_label="coord",
):
    output = StringIO()

    target_catalog.write(
        output,
        delimiter=" ",
        quotechar='"',
        format="ascii.fast_commented_header",
        overwrite=True,
        formats={
            "RA": lambda x: Angle(x, "deg")
            .wrap_at(360 * u.deg)
            .to_string("hr", sep=" ", precision=2),  # pylint: disable=E1101
            "Dec": lambda x: Angle(x, "deg").to_string("deg", sep=" ", precision=2),
            "Magnitude": "%.2f",
        },
    )

    content = output.getvalue().replace('"', "")
    output.close()

    with open(fn, "w") as fh:
        fh.write("LABEL " + host[host_id_label] + suffix + "\n")
        fh.write(
            "UTDATE  {yr} {mo:02} {day:02}\n".format(
                yr=obstime.datetime.year,
                mo=obstime.datetime.month,
                day=obstime.datetime.day,
            )
        )
        censtr = host[host_coord_label].to_string("hmsdms", sep=" ", precision=2, alwayssign=True)
        fh.write("CENTRE  " + censtr + "\n")
        fh.write("EQUINOX J2000.0\n")
        fh.write("# End of Header\n\n")
        fh.write(content)


def subsample_catalog(
    catalog,
    prilimits=None,
    maxflux=np.inf,
    maxguides=np.inf,
    maxsky=np.inf,
    verbose=True,
    favor_highpri=False,
):
    """
    Subsamples the catalog, limiting the number in a given priority set

    ``prilimits`` should be a dictionary mapping priority number to the maximum number of
    objects in that priority.  If it is None, no limits will be used.

    ``favor_highpri`` determines if the prilimits cuts pick favored (it must be an array
    which matches ``catalog``, where  *lower* numbers are favored), ora random sampling (False)
    """

    msks = [catalog["Notes"] == notestr for notestr in ("Flux", "Sky")]
    msks.append(np.array([n.lower().startswith("guide") for n in catalog["Notes"]]))
    maxns = [maxflux, maxsky, maxguides]

    if prilimits is not None:
        for pri, maxn in prilimits.items():
            msks.append(catalog["Priority"] == pri)
            maxns.append(maxn)

    idxs_to_rem = []
    for i, (msk, maxn) in enumerate(zip(msks, maxns)):
        ntorem = np.sum(msk) - maxn
        if ntorem >= 1:
            idxs = np.where(msk)[0]
            if favor_highpri is False or i < 3:
                idxs_to_rem.append(np.random.permutation(idxs)[:ntorem])
            else:
                assert len(favor_highpri) == len(catalog), "catalog and favor_highpri do not match in length!"
                idxs_to_rem.append(idxs[np.argsort(favor_highpri[idxs])][-ntorem:])
    subcat = catalog.copy()
    if idxs_to_rem:
        del subcat[np.concatenate(idxs_to_rem)]

    if verbose:
        npris = dict(enumerate(np.bincount(subcat["Priority"][subcat["Notes"] == "Targets"])))
        npris = {k: v for k, v in npris.items() if v > 0}
        print(
            "Nflux:",
            np.sum(subcat["Notes"] == "Flux"),
            "Nguide:",
            np.sum([n.lower().startswith("guide") for n in subcat["Notes"]]),
            "NSky",
            np.sum(subcat["Notes"] == "Sky"),
        )
        print("Targets in each priority:", npris)
        print("Total:", len(subcat))

    return subcat


def infer_radec_cols(table):
    raname = decname = None
    for cnm in table.colnames:
        if cnm.lower() == "ra":
            raname = cnm
        elif cnm.lower() == "dec":
            decname = cnm
    return raname, decname


def make_decals_viewer_cutouts(table, survey="sdss", ncols=3, zoom=15, size=120, namecol=None, dhtml=True):
    """
    Zoom of 15 is ~1"/pixel, so ~2' across with defaults
    """
    template_url = (
        "http://legacysurvey.org/viewer/jpeg-cutout/?ra={ra:.7}&dec={dec:.7}&zoom={zoom}&layer={layer}&size={size}"
    )

    raname, decname = infer_radec_cols(table)

    entries = []
    for row in table:
        imgurl = template_url.format(ra=row[raname], dec=row[decname], layer=survey, size=size, zoom=zoom)
        viewurl = "http://legacysurvey.org/viewer?ra={}&dec={}".format(row[raname], row[decname])

        namestr = "" if namecol is None else (str(row[namecol]) + "<br>")
        entries.append('{}<a href="{}"><img src="{}"></a>'.format(namestr, viewurl, imgurl))

    entryrows = [[]]
    while entries:
        entry = entries.pop(0)
        if len(entryrows[-1]) >= ncols:
            entryrows.append([])
        entryrows[-1].append(entry)
    entryrows[-1].extend([""] * (ncols - len(entryrows[-1])))

    tabrows = ["<td>{}</td>".format("</td><td>".join(erow)) for erow in entryrows]

    htmlstr = """
    <table>
    <tr>{}</tr>
    </table>
    """.format(
        "</tr>\n<tr>".join(tabrows)
    )

    if dhtml:
        from IPython import display

        return display.HTML(htmlstr)
    else:
        return htmlstr


def make_des_cutouts_file(table, copytoclipboard=True, showtable=False):
    raname, decname = infer_radec_cols(table)

    entries = []
    for row in table:
        entries.append("{:.5f},{:.5f}".format(row[raname], row[decname]))

    htmlstr = """
    Paste the coords in "Enter values" at
    <a href="https://des.ncsa.illinois.edu/easyweb/cutouts">the DES cutout service</a>,
    and change the size to 0.3 arcmin
    """
    tablestr = """<br>
    <table>
    <tr><td>{}</td></tr>
    </table>
    """.format(
        "</td></tr>\n<tr><td>".join(entries)
    )
    if showtable:
        htmlstr += tablestr

    if copytoclipboard:
        import platform

        text = "\n".join(entries)
        if platform.system() == "Darwin":
            clipproc = os.popen("pbcopy", "w")
            clipproc.write(text)
            clipproc.close()
            print("Copied to clipboard")
        elif platform.system() == "Linux":
            clipproc = os.popen("xsel -i", "w")
            clipproc.write(text)
            clipproc.close()
            print("Copied to clipboard")
        else:
            raise OSError("Not on a mac or linux, so can't use clipboard. ")

    from IPython import display

    return display.HTML(htmlstr)


def show_des_cutouts(
    table,
    jobname,
    username="eteq",
    ncols=3,
    namecol=None,
    force_size=(128, 128),
    dhtml=True,
):
    base_url = "https://des.ncsa.illinois.edu"

    list_url = base_url + "/easyweb/static/workdir/{}/{}/list.json".format(username, jobname)
    print(list_url)
    list_json = requests.get(list_url).json()
    img_urls = [base_url + img["name"] for img in list_json]

    if len(table) != len(img_urls):
        raise ValueError("table and job length do not match!")
    raname, decname = infer_radec_cols(table)

    sizestr = "" if force_size is None else ('height="{}" width="{}"'.format(force_size[1], force_size[0]))
    entries = []
    for row, imgurl in zip(table, img_urls):
        viewurl = "http://legacysurvey.org/viewer?ra={}&dec={}".format(row[raname], row[decname])

        namestr = "" if namecol is None else (str(row[namecol]) + "<br>")
        entries.append('{}<a href="{}"><img src="{}"{}></a>'.format(namestr, viewurl, imgurl, sizestr))

    entryrows = [[]]
    while entries:
        entry = entries.pop(0)
        if len(entryrows[-1]) >= ncols:
            entryrows.append([])
        entryrows[-1].append(entry)
    entryrows[-1].extend([""] * (ncols - len(entryrows[-1])))

    tabrows = ["<td>{}</td>".format("</td><td>".join(erow)) for erow in entryrows]

    htmlstr = """
    <table>
    <tr>{}</tr>
    </table>
    """.format(
        "</tr>\n<tr>".join(tabrows)
    )

    if dhtml:
        from IPython import display

        return display.HTML(htmlstr)
    else:
        return htmlstr
