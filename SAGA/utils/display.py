from IPython.display import display_html

__all__ = ["print_for_viewer", "show_images"]


def print_for_viewer(table, keys=("OBJID", "RA", "DEC"), additional_keys=None):
    keys = list(keys)
    if additional_keys:
        keys.extend(additional_keys)
    table[keys].pprint(-1, -1)


def show_images(
    table,
    pixscale=0.2,
    size=200,
    keys=("OBJID", "RA", "DEC", "HOSTID", "r_mag", "sb_r", "gr", "SPEC_Z", "ZQUALITY", "TELNAME"),
    layer="dr8",
    ra_label="RA",
    dec_label="DEC",
    use_dev=False,
    additional_keys=None,
):
    out = []
    dev = "-dev" if use_dev else ""
    keys_used = [key for key in keys if key in table.colnames] if keys else []
    if additional_keys:
        keys_used.extend(additional_keys)
    if dec_label not in keys_used:
        keys_used = [dec_label] + keys_used
    if ra_label not in keys_used:
        keys_used = [ra_label] + keys_used
    for row in table:
        if layer == "auto":
            if row["survey"] == "des":
                layer = "des-dr1"
            elif row["survey"] == "decals":
                layer = "ls-dr67"
            else:
                layer = "sdss2"

        url = "http://legacysurvey.org/viewer{dev}/jpeg-cutout/?ra={ra}&dec={dec}&pixscale={pixscale}&layer={layer}&size={size}".format(
            ra=row[ra_label],
            dec=row[dec_label],
            layer=layer,
            pixscale=pixscale,
            size=size,
            dev=dev,
        )

        link = "http://legacysurvey.org/viewer{dev}?ra={ra}&dec={dec}&layer={layer}&zoom=16".format(
            ra=row[ra_label], dec=row[dec_label], layer=layer, dev=dev,
        )
        title = "\n".join((f"{k} = {row[k]}" for k in keys_used))

        out.append(
            f'<a href="{link}" target="_blank"><img src="{url}" style="display:inline-block; width:{size}px; height:{size}px" title="{title}" /></a>'
        )
    display_html("\n".join(out), raw=True)
