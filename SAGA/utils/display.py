from IPython.display import display_html


def show_images(
    table,
    pixscale=0.2,
    size=200,
    keys=("OBJID", "RA", "DEC", "HOSTID", "survey", "r_mag", "sb_r", "gr", "RHOST_KPC"),
    layer="auto",
    ra_label="RA",
    dec_label="DEC",
):
    out = []
    for row in table:
        if layer == "auto":
            if row["survey"] == "des":
                layer = "des-dr1"
            elif row["survey"] == "decals":
                layer = "dr8"
            else:
                layer = "sdss2"

        url = "http://legacysurvey.org/viewer-dev/jpeg-cutout/?ra={ra}&dec={dec}&pixscale={pixscale}&layer={layer}&size={size}".format(
            ra=row[ra_label], dec=row[dec_label], layer=layer, pixscale=pixscale, size=size,
        )

        link = "http://legacysurvey.org/viewer-dev?ra={ra}&dec={dec}&layer={layer}&zoom=16".format(
            ra=row[ra_label], dec=row[dec_label], layer=layer,
        )
        title = "\n".join(("{} = {}".format(k, row[k]) for k in keys))

        out.append(
            '<a href="{}" target="_blank"><img src="{}" style="display:inline-block;" title="{}" /></a>'.format(
                link, url, title
            )
        )
    display_html("\n".join(out), raw=True)
